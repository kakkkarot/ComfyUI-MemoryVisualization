WEB_DIRECTORY = "web"
NODE_CLASS_MAPPINGS = {}

import logging
import asyncio
import torch
import server
from aiohttp import web
import psutil
import comfy.model_management
import comfy.memory_management


log = logging.getLogger(__name__)

try:
    import comfy_aimdo.control
except ImportError:
    comfy_aimdo = None

# NVML handle + power-cap cache. Cap is static for a given driver state,
# so we only query it once. Handle init is best-effort; failures stick.
_nvml_state = {"handle": None, "tried": False, "power_limit": None}

def _nvml_handle(device):
    if _nvml_state["tried"]:
        return _nvml_state["handle"]
    _nvml_state["tried"] = True
    try:
        import pynvml
        pynvml.nvmlInit()
        idx = device.index if device.index is not None else 0
        _nvml_state["handle"] = pynvml.nvmlDeviceGetHandleByIndex(idx)
    except Exception as e:
        log.debug("aimdo-viz: pynvml init failed: %s", e)
    return _nvml_state["handle"]

def _nvml_power_limit(device):
    if _nvml_state["power_limit"] is not None:
        return _nvml_state["power_limit"]
    h = _nvml_handle(device)
    if h is None:
        return None
    try:
        import pynvml
        _nvml_state["power_limit"] = pynvml.nvmlDeviceGetPowerManagementLimit(h)
        return _nvml_state["power_limit"]
    except Exception as e:
        log.debug("aimdo-viz: nvmlDeviceGetPowerManagementLimit failed: %s", e)
        return None

def _get_lock():
    # Stored on comfy.model_management so the same lock survives hot reloads.
    mm = comfy.model_management
    if not hasattr(mm, '_viz_model_lock'):
        mm._viz_model_lock = asyncio.Lock()
    return mm._viz_model_lock
routes = server.PromptServer.instance.routes

@routes.get("/aimdo/vram")
async def aimdo_vram_status(request):
    device = comfy.model_management.get_torch_device()
    if not torch.cuda.is_available() or device.type != "cuda":
        return web.json_response({"enabled": False})

    aimdo_active = getattr(comfy.memory_management, 'aimdo_enabled', False) and comfy_aimdo is not None

    models = []
    loaded_models = list(comfy.model_management.current_loaded_models)
    for model_idx, lm in enumerate(loaded_models):
        patcher = lm.model
        if patcher is None:
            continue

        model_obj = patcher.model
        if model_obj is None:
            continue

        name = model_obj.__class__.__name__
        is_dynamic = patcher.is_dynamic()
        total_size = patcher.model_size()
        loaded = patcher.loaded_size()

        # RAM side: pinned host memory used for fast transfers, plus
        # non-pinned loaded host memory when the patcher exposes it.
        pinned_ram = 0
        try:
            if hasattr(patcher, 'pinned_memory_size'):
                pinned_ram = patcher.pinned_memory_size()
        except Exception as e:
            log.debug("aimdo-viz: pinned_memory_size failed: %s", e)
        loaded_ram = 0
        try:
            if hasattr(patcher, 'loaded_ram_size'):
                loaded_ram = max(0, patcher.loaded_ram_size() - pinned_ram)
        except Exception as e:
            log.debug("aimdo-viz: loaded_ram_size failed: %s", e)

        # VBAR state per device (aimdo only)
        vbars = []
        vbar_loaded_total = 0

        if aimdo_active and is_dynamic and hasattr(model_obj, "dynamic_vbars"):
            for dev, vbar in model_obj.dynamic_vbars.items():
                try:
                    loaded_bytes = vbar.loaded_size()
                    vbar_loaded_total += loaded_bytes
                    full_residency = vbar.get_residency()
                    page_size = getattr(vbar, 'page_size', 32 * 1024 * 1024)
                    vbar_offset = getattr(vbar, 'offset', 0)
                    if vbar_offset > 0:
                        used_pages = (vbar_offset + page_size - 1) // page_size
                    else:
                        used_pages = (total_size + page_size - 1) // page_size
                    vbars.append({
                        "device": str(dev),
                        "loaded": loaded_bytes,
                        "watermark": vbar.get_watermark(),
                        "residency": full_residency[:used_pages],
                    })
                except Exception as e:
                    log.warning("aimdo-viz: VBAR query failed: %s", e)

        entry = {
            "index": model_idx,
            "name": name,
            "total_size": total_size,
            "loaded_size": loaded,
            "vbar_loaded": vbar_loaded_total,
            "ram_size": max(0, total_size - vbar_loaded_total),
            "pinned_ram": pinned_ram,
            "loaded_ram": loaded_ram,
            "dynamic": is_dynamic,
            "vbars": vbars,
        }

        models.append(entry)

    has_dynamic = any(m.get("dynamic") for m in models)
    aimdo_usage = comfy_aimdo.control.get_total_vram_usage() if aimdo_active and has_dynamic else 0

    # driver-level free/total (matches nvitop)
    free_cuda, total_vram = torch.cuda.mem_get_info(device)

    try:
        gpu_util = torch.cuda.utilization(device)
    except Exception:
        gpu_util = None

    try:
        gpu_temp = torch.cuda.temperature(device)
    except Exception:
        gpu_temp = None

    try:
        gpu_power = torch.cuda.power_draw(device)  # mW
    except Exception:
        gpu_power = None
    gpu_power_limit = _nvml_power_limit(device)  # mW

    # pytorch internal stats
    stats = torch.cuda.memory_stats(device)
    torch_active = stats.get('active_bytes.all.current', 0)
    torch_reserved = stats.get('reserved_bytes.all.current', 0)

    ram = psutil.virtual_memory()
    process_ram = psutil.Process().memory_info().rss
    total_pinned = sum(m.get("pinned_ram", 0) for m in models)
    total_loaded_ram = sum(m.get("loaded_ram", 0) for m in models)

    return web.json_response({
        "enabled": True,
        "aimdo_active": aimdo_active,
        "total_vram": total_vram,
        "free_vram": free_cuda,
        "gpu_util": gpu_util,
        "gpu_temp": gpu_temp,
        "gpu_power": gpu_power,
        "gpu_power_limit": gpu_power_limit,
        "aimdo_usage": aimdo_usage,
        "torch_active": torch_active,
        "torch_reserved": torch_reserved,
        "total_ram": ram.total,
        "used_ram": ram.used,
        "process_ram": process_ram,
        "pinned_ram": total_pinned,
        "loaded_ram": total_loaded_ram,
        "models": models,
    })

@routes.post("/aimdo/unload_all")
async def aimdo_unload_all(request):
    if _is_executing():
        return web.json_response({"error": "cannot unload during execution"}, status=409)
    async with _get_lock():
        await asyncio.get_running_loop().run_in_executor(None, comfy.model_management.unload_all_models)
    return web.json_response({"status": "ok"})


def _is_executing():
    return bool(server.PromptServer.instance.prompt_queue.currently_running)

def _get_model_idx(data):
    idx = data.get("index")
    if not isinstance(idx, int) or isinstance(idx, bool):
        return None, web.json_response({"error": "missing or invalid index"}, status=400)
    models = comfy.model_management.current_loaded_models
    if idx < 0 or idx >= len(models):
        return None, web.json_response({"error": "index out of range"}, status=400)
    return idx, None

@routes.post("/aimdo/reset_watermark")
async def aimdo_reset_watermark(request):
    idx, err = _get_model_idx(await request.json())
    if err:
        return err
    async with _get_lock():
        torch.cuda.empty_cache()
        models = comfy.model_management.current_loaded_models
        if idx >= len(models):
            return web.json_response({"error": "model no longer at index"}, status=409)
        patcher = models[idx].model
        if patcher is not None and hasattr(patcher, '_vbar_get'):
            vbar = patcher._vbar_get()
            if vbar is not None:
                vbar.prioritize()
    return web.json_response({"status": "ok"})

@routes.post("/aimdo/unload_model")
async def aimdo_unload_model(request):
    if _is_executing():
        return web.json_response({"error": "cannot unload during execution"}, status=409)
    idx, err = _get_model_idx(await request.json())
    if err:
        return err
    async with _get_lock():
        models = comfy.model_management.current_loaded_models
        if idx >= len(models):
            return web.json_response({"error": "model no longer at index"}, status=409)
        models[idx].model_unload()
        models.pop(idx)
        comfy.model_management.soft_empty_cache()
    return web.json_response({"status": "ok"})
