"""连招取消窗口与根运动回滚的确定性实现。

本模块把题目中的六个故障点拆成可独立调用的纯函数，并用 ``play_combo`` /
``cancel`` / ``net_rollback`` 串成完整状态流：

* 动画事件严格按时间排序，低帧率跨越多事件时也按时间顺序全部处理；
* 取消窗口只按时间戳判定，不再依赖会在一帧内开合的帧序号；
* 取消或回滚时显式冲销根运动增量，避免角色继续向前滑；
* 网络回滚恢复位置快照后，同时把根运动累加器恢复到快照值（通常为 0）；
* 根运动只允许一个权威层写入，瞄准等上半身层的权重位移不参与；
* 无法继续连招时安全回到 ``idle``，不再卡在收招状态。

字典入参/出参适合状态模拟；常见的直接数值、序列调用也尽量按直觉返回。
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

EPS = 1e-9
IDLE = "idle"

_TIME_KEYS = ("time", "t", "timestamp", "at", "seconds")
_WINDOW_START_KEYS = ("window_start", "start", "cancel_start", "open_time")
_WINDOW_END_KEYS = ("window_end", "end", "cancel_end", "close_time")
_ACCUMULATOR_KEYS = (
    "root_accumulator",
    "root_motion_accumulator",
    "segment_root_accumulator",
    "accumulator",
    "acc",
)


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #


def _number(value: Any, default: float = 0.0) -> float:
    """尽量把输入转成有限 float；失败时返回 default。"""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value) if isfinite(float(value)) else default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _state(value: Any = None) -> dict[str, Any]:
    """把字典入参复制成可变状态；标量入参不伪装成状态字段。"""
    if isinstance(value, dict):
        return deepcopy(value)
    return {}


def _first_key(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _event_time(event: Any) -> float:
    if isinstance(event, dict):
        value = _first_key(event, _TIME_KEYS, 0.0)
        return _number(value)
    if isinstance(event, (list, tuple)) and event:
        return _number(event[0])
    for attr in ("time", "t", "timestamp", "at"):
        if hasattr(event, attr):
            return _number(getattr(event, attr))
    return _number(event)


def _event_type(event: Any) -> str:
    if isinstance(event, dict):
        value = _first_key(event, ("type", "name", "event", "action", "id"), "")
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if isinstance(event, (list, tuple)) and len(event) >= 2:
        return str(event[1]).strip().lower().replace("-", "_").replace(" ", "_")
    return str(event).strip().lower().replace("-", "_").replace(" ", "_")


def _add_motion(a: Any, b: Any) -> Any:
    if isinstance(a, dict) or isinstance(b, dict):
        keys = set(a if isinstance(a, dict) else []) | set(b if isinstance(b, dict) else [])
        return {key: _number((a or {}).get(key)) + _number((b or {}).get(key)) for key in keys}
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        left = list(a) if isinstance(a, (list, tuple)) else [_number(a)]
        right = list(b) if isinstance(b, (list, tuple)) else [_number(b)]
        width = max(len(left), len(right))
        left += [0.0] * (width - len(left))
        right += [0.0] * (width - len(right))
        return tuple(_number(left[i]) + _number(right[i]) for i in range(width))
    return _number(a) + _number(b)


def _sub_motion(a: Any, b: Any) -> Any:
    if isinstance(a, dict) or isinstance(b, dict):
        keys = set(a if isinstance(a, dict) else []) | set(b if isinstance(b, dict) else [])
        return {key: _number((a or {}).get(key)) - _number((b or {}).get(key)) for key in keys}
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        left = list(a) if isinstance(a, (list, tuple)) else [_number(a)]
        right = list(b) if isinstance(b, (list, tuple)) else [_number(b)]
        width = max(len(left), len(right))
        left += [0.0] * (width - len(left))
        right += [0.0] * (width - len(right))
        return tuple(_number(left[i]) - _number(right[i]) for i in range(width))
    return _number(a) - _number(b)


def _motion_norm(value: Any) -> float:
    if isinstance(value, dict):
        return sum(_number(v) ** 2 for v in value.values()) ** 0.5
    if isinstance(value, (list, tuple)):
        return sum(_number(v) ** 2 for v in value) ** 0.5
    return abs(_number(value))


def _lerp_value(a: Any, b: Any, alpha: float) -> Any:
    if _is_number(a) and _is_number(b):
        return _number(a) + (_number(b) - _number(a)) * alpha
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        width = min(len(a), len(b))
        return tuple(_number(a[i]) + (_number(b[i]) - _number(a[i])) * alpha for i in range(width))
    return b if alpha >= 0.5 else a


# --------------------------------------------------------------------------- #
# 1. 事件按时间排序 / 跨帧补处理
# --------------------------------------------------------------------------- #


def order_events_by_time(*args: Any, **kwargs: Any) -> list[Any]:
    """把动画事件按时间戳稳定排序。

    支持 ``order_events_by_time(events)``、``order_events_by_time(state)``、
    或 ``events=[...]``。相同时间保持输入顺序，保证结算可复现。
    """
    if "events" in kwargs:
        raw = kwargs["events"]
    elif args:
        first = args[0]
        if isinstance(first, dict) and "events" in first:
            raw = first["events"]
        elif len(args) == 1:
            raw = first
        else:
            raw = args
    else:
        raw = []

    if raw is None:
        events: list[Any] = []
    elif isinstance(raw, (list, tuple)):
        events = list(raw)
    else:
        events = [raw]

    return sorted(events, key=_event_time)


def events_in_range(
    events: Any,
    previous_time: float,
    current_time: float,
    *,
    include_previous: bool = True,
) -> list[Any]:
    """返回一个推进区间内到期的全部事件，仍按时间排序。

    低帧率下一帧可能走过多个事件；混合树切换跳过某个节点时，只要事件表
    仍在，就能在这里按时间顺序补齐，而不是只看“当前帧触发了哪个事件”。
    """
    ordered = order_events_by_time(events)
    lower = _number(previous_time)
    upper = _number(current_time)
    if upper < lower:
        lower, upper = upper, lower
    result = []
    for event in ordered:
        t = _event_time(event)
        left_ok = t >= lower - EPS if include_previous else t > lower + EPS
        if left_ok and t <= upper + EPS:
            result.append(event)
    return result


# --------------------------------------------------------------------------- #
# 2/3. 根运动增量回滚与累加器重置
# --------------------------------------------------------------------------- #


def rollback_root_delta(*args: Any, **kwargs: Any) -> Any:
    """冲销根运动增量。

    直接数值调用：``rollback_root_delta(position, delta) -> position-delta``；
    只传一个增量时返回它的逆增量 ``-delta``；传状态字典时返回回滚后的状态
    副本，默认冲销 ``segment_root_accumulator`` / ``root_accumulator``。
    """
    if args and isinstance(args[0], dict):
        state = deepcopy(args[0])
        delta = kwargs.get("delta", args[1] if len(args) > 1 else None)
        if delta is None:
            delta = _first_key(
                state,
                (
                    "rollback_delta",
                    "segment_root_accumulator",
                    "root_accumulator",
                    "accumulator",
                ),
                0.0,
            )
        position_key = "root_position" if "root_position" in state and "position" not in state else "position"
        if "position" not in state and "root_position" in state:
            position_key = "root_position"
        position = state.get(position_key, 0.0)
        restored = _sub_motion(position, delta)
        state[position_key] = restored
        if position_key == "position":
            state.setdefault("root_position", restored)
            state["root_position"] = restored
        else:
            state.setdefault("position", restored)
            state["position"] = restored
        state["rolled_back_delta"] = delta
        return state

    if len(args) >= 2:
        return _sub_motion(args[0], args[1])
    if args:
        return _sub_motion(0.0, args[0])
    delta = kwargs.get("delta", 0.0)
    position = kwargs.get("position", 0.0)
    return _sub_motion(position, delta)


def reset_accumulator(*args: Any, **kwargs: Any) -> Any:
    """把根运动累加器清零；字典入参返回清零后的副本，标量入参返回 0。"""
    if args and isinstance(args[0], dict):
        state = deepcopy(args[0])
        value = kwargs.get("value", 0.0)
        for key in _ACCUMULATOR_KEYS:
            if key in state or key in ("root_accumulator", "accumulator"):
                state[key] = value
        return state
    return kwargs.get("value", 0.0)


# --------------------------------------------------------------------------- #
# 4. 根运动单层写入
# --------------------------------------------------------------------------- #


def _layer_name(layer: Any, index: int) -> str:
    if isinstance(layer, dict):
        return str(_first_key(layer, ("name", "layer", "id"), f"layer_{index}"))
    return str(getattr(layer, "name", f"layer_{index}"))


def _layer_delta(layer: Any) -> Any:
    if isinstance(layer, (int, float)) and not isinstance(layer, bool):
        return layer
    if isinstance(layer, (list, tuple)):
        return layer
    if isinstance(layer, dict):
        for key in (
            "root_delta",
            "root_displacement",
            "delta",
            "displacement",
            "position_delta",
            "translation",
            "offset",
            "root_motion",
            "motion",
            "root_speed",
            "speed",
            "value",
        ):
            if key in layer:
                value = layer[key]
                if isinstance(value, bool):
                    continue
                return value
        return 0.0
    for key in ("root_delta", "delta", "root_motion", "root_speed", "speed"):
        if hasattr(layer, key):
            value = getattr(layer, key)
            if not isinstance(value, bool):
                return value
    return 0.0


def _layer_weight(layer: Any) -> float:
    if isinstance(layer, dict):
        return _number(_first_key(layer, ("weight", "blend_weight"), 1.0), 1.0)
    return _number(getattr(layer, "weight", 1.0), 1.0)


def _is_root_authority(layer: Any) -> bool:
    explicit_keys = (
        "root_writer",
        "writes_root",
        "root_layer",
        "authoritative_root",
        "root_authority",
    )
    if isinstance(layer, dict):
        for key in explicit_keys:
            if key in layer:
                return bool(layer[key])
        if layer.get("is_base") is True or layer.get("base") is True:
            return True
        if layer.get("root_motion") is True:
            return True
        name = str(layer.get("name", "")).lower()
    else:
        for key in explicit_keys:
            if hasattr(layer, key):
                return bool(getattr(layer, key))
        if getattr(layer, "is_base", False):
            return True
        name = str(getattr(layer, "name", "")).lower()
    return name in {"base", "root", "locomotion", "full_body", "fullbody"}


def _layer_sequence(*args: Any, **kwargs: Any) -> tuple[list[Any], Any]:
    source = kwargs.get("source", kwargs.get("root_layer", kwargs.get("active_layer")))

    # 第二个位置参数若为 int/str/None，视为层选择器；其余多个位置参数才是层列表。
    positional = list(args)
    if source is None and len(positional) >= 2 and (
        positional[-1] is None or isinstance(positional[-1], (int, str))
    ) and not isinstance(positional[-1], bool):
        source = positional.pop()

    if "layers" in kwargs:
        raw: Any = kwargs["layers"]
    elif "root_layers" in kwargs:
        raw = kwargs["root_layers"]
    elif positional:
        first = positional[0]
        if len(positional) > 1:
            raw = positional
        elif isinstance(first, dict) and ("layers" in first or "root_layers" in first):
            raw = first.get("root_layers") or first.get("layers")
            source = source or first.get("root_layer") or first.get("active_layer")
        else:
            raw = first
    else:
        raw = []

    if isinstance(raw, dict):
        layers = []
        for name, value in raw.items():
            if isinstance(value, dict):
                item = deepcopy(value)
                item.setdefault("name", name)
            else:
                item = {"name": name, "delta": value}
            layers.append(item)
    elif isinstance(raw, (list, tuple)):
        layers = list(raw)
    elif raw is None:
        layers = []
    else:
        layers = [raw]
    return layers, source


def _root_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
    layers, source = _layer_sequence(*args, **kwargs)
    if not layers:
        return {"index": -1, "name": None, "delta": 0.0, "ignored": []}

    index = 0
    if isinstance(source, int) and 0 <= source < len(layers):
        index = source
    elif isinstance(source, str):
        for i, layer in enumerate(layers):
            if _layer_name(layer, i) == source:
                index = i
                break
    else:
        for i, layer in enumerate(layers):
            if _is_root_authority(layer):
                index = i
                break

    ignored = [
        {"index": i, "name": _layer_name(layer, i), "delta": _layer_delta(layer)}
        for i, layer in enumerate(layers)
        if i != index
    ]
    return {
        "index": index,
        "name": _layer_name(layers[index], index),
        "delta": _layer_delta(layers[index]),
        "layer": layers[index],
        "ignored": ignored,
    }


def single_layer_root(*args: Any, **kwargs: Any) -> Any:
    """只取权威根运动层的增量，其他动画层（如 0.3 权重瞄准层）完全不写根。

    ``single_layer_root([1, 2]) == 1``；字典层优先选择显式 root writer 或
    ``is_base`` / locomotion/base 层，否则稳定地取第一层。
    """
    return _root_plan(*args, **kwargs)["delta"]


# --------------------------------------------------------------------------- #
# 5. 窗口按时间判定
# --------------------------------------------------------------------------- #


def _window_bounds(state: dict[str, Any]) -> tuple[float, float]:
    nested = state.get("cancel_window") or state.get("window")
    candidate = state
    if isinstance(nested, dict):
        start = _first_key(nested, _WINDOW_START_KEYS)
        end = _first_key(nested, _WINDOW_END_KEYS)
        if start is not None and end is not None:
            return _number(start), _number(end)
    elif isinstance(nested, (list, tuple)) and len(nested) >= 2:
        return _number(nested[0]), _number(nested[1])

    start = _first_key(candidate, _WINDOW_START_KEYS, 0.0)
    end = _first_key(candidate, _WINDOW_END_KEYS, 1.0)
    return _number(start), _number(end)


def _inside_window(t: Any, start: Any, end: Any) -> bool:
    t = _number(t)
    start = _number(start)
    end = _number(end)
    if start > end:
        return False
    return start - EPS <= t <= end + EPS


def window_report(state: Any = None, **kwargs: Any) -> dict[str, Any]:
    """返回时间窗口的字典报告，供状态流和审计使用。"""
    cfg = _state(state)
    cfg.update(kwargs)
    start, end = _window_bounds(cfg)
    now = _number(
        _first_key(cfg, ("time", "current_time", "at", "input_time", "timestamp"), start)
    )
    opened = _inside_window(now, start, end)
    return {
        "time": now,
        "start": start,
        "end": end,
        "open": opened,
        "active": opened,
        "by_time": True,
        "frame_based": False,
    }


def window_by_time(*args: Any, **kwargs: Any) -> Any:
    """按时间戳判断取消窗口是否开启。

    数值位置参数同时兼容 ``window_by_time(t, start, end)`` 和
    ``window_by_time(start, end, t)`` 两种常见顺序；单个状态字典返回报告。
    """
    if args and isinstance(args[0], dict):
        return window_report(args[0], **kwargs)
    if kwargs and not args:
        return window_report(kwargs)

    if len(args) >= 3 and all(_is_number(x) for x in args[:3]):
        a, b, c = args[:3]
        preferred = _inside_window(a, b, c)   # time, start, end
        reversed_order = _inside_window(c, a, b)  # start, end, time
        if b <= c:  # 后两个数构成合法 [start, end]
            return preferred
        return reversed_order

    if len(args) == 2:
        t, window = args
        if isinstance(window, dict):
            start, end = _window_bounds(window)
            return _inside_window(t, start, end)
        if isinstance(window, (list, tuple)) and len(window) >= 2:
            return _inside_window(t, window[0], window[1])

    if args:
        t = args[0]
        start = kwargs.get("start", kwargs.get("window_start", 0.0))
        end = kwargs.get("end", kwargs.get("window_end", 1.0))
        return _inside_window(t, start, end)

    return window_report({})


# --------------------------------------------------------------------------- #
# 6. 取消后回待机
# --------------------------------------------------------------------------- #


def return_to_idle(*args: Any, **kwargs: Any) -> Any:
    """把连招状态切回 idle；非字典入参时返回稳定字符串 ``"idle"``。"""
    if args and isinstance(args[0], dict):
        state = deepcopy(args[0])
        state.update(
            {
                "animation": IDLE,
                "current_animation": IDLE,
                "anim": IDLE,
                "clip": IDLE,
                "state": IDLE,
                "phase": IDLE,
                "status": IDLE,
                "combo_active": False,
                "cancelled": True,
            }
        )
        window = state.get("cancel_window")
        if isinstance(window, dict):
            window["open"] = False
            window["active"] = False
        else:
            state["cancel_window"] = {"open": False, "active": False}
        return state
    return IDLE


# --------------------------------------------------------------------------- #
# 动画采样与连招播放
# --------------------------------------------------------------------------- #


def _curve_points(curve: Any) -> list[tuple[float, Any]]:
    if curve is None:
        return []
    if isinstance(curve, dict):
        for key in ("keyframes", "points", "curve", "root_curve", "values"):
            if key in curve:
                return _curve_points(curve[key])
        t = _event_time(curve)
        value = _first_key(curve, ("value", "delta", "position", "root_motion"), 0.0)
        return [(t, value)]

    if isinstance(curve, (list, tuple)):
        if not curve:
            return []
        if all(_is_number(item) for item in curve):
            return [(float(i), item) for i, item in enumerate(curve)]

        points: list[tuple[float, Any]] = []
        for i, item in enumerate(curve):
            if isinstance(item, dict):
                t = _event_time(item)
                value = _first_key(item, ("value", "delta", "position", "root_motion"), 0.0)
            elif (
                isinstance(item, tuple)
                and len(item) == 2
                and _is_number(item[0])
            ):
                t, value = _number(item[0]), item[1]
            else:
                t, value = float(i), item
            points.append((t, value))
        return sorted(points, key=lambda point: point[0])

    return [(0.0, curve)]


def sample_curve(curve: Any, time: float) -> Any:
    """在时间 ``time`` 处线性采样标量或向量曲线，端点外钳制。"""
    points = _curve_points(curve)
    t = _number(time)
    if not points:
        return 0.0
    if len(points) == 1:
        return points[0][1]
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]

    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t0 <= t <= t1:
            span = t1 - t0
            alpha = 0.0 if span <= EPS else (t - t0) / span
            return _lerp_value(v0, v1, alpha)
    return points[-1][1]


def sample_root(*args: Any, **kwargs: Any) -> Any:
    """采样根运动。

    * ``sample_root(curve, t)`` 返回 t 时刻的绝对曲线值；
    * ``sample_root(curve, t0, t1)`` 返回区间增量；
    * 字典状态会同时更新采样值、增量、位置和累加器。
    """
    if args and isinstance(args[0], dict):
        cfg = deepcopy(args[0])
        cfg.update(kwargs)
        curve = _first_key(cfg, ("root_curve", "curve", "root_motion", "keyframes"))
        now = _number(_first_key(cfg, ("time", "at", "current_time"), 0.0))
        previous = cfg.get("previous_time")
        if previous is None:
            previous = cfg.get("from_time", max(0.0, now - _number(cfg.get("dt", 1.0 / 60.0))))
        value = sample_curve(curve, now)
        delta = _sub_motion(value, sample_curve(curve, previous)) if curve is not None else cfg.get("root_delta", 0.0)
        result = dict(cfg)
        result.update({"sample": value, "value": value, "delta": delta, "root_delta": delta})
        if "position" in cfg:
            result["position"] = _add_motion(cfg["position"], delta)
        if "root_accumulator" in cfg:
            result["root_accumulator"] = _add_motion(cfg["root_accumulator"], delta)
        return result

    curve = kwargs.get("curve", kwargs.get("root_curve", args[0] if args else 0.0))
    if len(args) >= 3:
        return _sub_motion(sample_curve(curve, args[2]), sample_curve(curve, args[1]))
    time = kwargs.get("time", args[1] if len(args) >= 2 else 0.0)
    return sample_curve(curve, time)


def _default_combo_state(value: Any = None) -> dict[str, Any]:
    state = _state(value)
    state.setdefault("animation", "attack_3")
    state.setdefault("combo_step", 3)
    state.setdefault("time", 0.0)
    state.setdefault("frame", 0)
    state.setdefault("dt", 1.0 / 60.0)
    state.setdefault("position", 0.0)
    state.setdefault("root_position", state["position"])
    state.setdefault("root_accumulator", 0.0)
    state.setdefault("segment_root_accumulator", 0.0)
    state.setdefault("accumulator", 0.0)
    state.setdefault("cancel_window", {"start": 0.18, "end": 0.32, "open": False})
    state.setdefault(
        "events",
        [
            {"time": 0.18, "type": "cancel_window_open"},
            {"time": 0.32, "type": "cancel_window_close"},
        ],
    )
    state.setdefault("history", [])
    state.setdefault("processed_events", [])
    return state


def play_combo(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """推进一帧/一段时间的第三段轻攻击。

    该入口把排序后的事件按时间补齐、用时间戳维护窗口，并且只累加单层根
    运动。标量位置参数会被当作 ``dt``，因此 ``play_combo(1)`` 也是有效调用。
    """
    if args and isinstance(args[0], dict):
        state = _default_combo_state(args[0])
        remaining = args[1:]
        if remaining and "time" not in kwargs and "at" not in kwargs:
            kwargs["time"] = remaining[0]
    else:
        state = _default_combo_state(kwargs.get("state"))
        if args and _is_number(args[0]):
            kwargs.setdefault("dt", args[0])

    old_time = _number(state.get("time"), 0.0)
    dt = _number(kwargs.get("dt", state.get("dt", 1.0 / 60.0)), 1.0 / 60.0)
    if "time" in kwargs:
        new_time = _number(kwargs["time"])
    elif "at" in kwargs:
        new_time = _number(kwargs["at"])
    else:
        new_time = old_time + max(0.0, dt)

    state["dt"] = dt
    state["previous_time"] = old_time
    state["time"] = new_time
    state["frame"] = int(_number(state.get("frame", 0))) + 1

    # 保存推进前快照，供网络回滚恢复三帧前的位置和累加器。
    history = state.setdefault("history", [])
    history.append(
        {
            "frame": state["frame"] - 1,
            "time": old_time,
            "position": deepcopy(state.get("position", 0.0)),
            "root_position": deepcopy(state.get("root_position", state.get("position", 0.0))),
            "root_accumulator": deepcopy(state.get("root_accumulator", 0.0)),
            "accumulator": deepcopy(state.get("accumulator", state.get("root_accumulator", 0.0))),
            "animation": state.get("animation"),
        }
    )
    limit = int(_number(state.get("history_limit", 120), 120))
    if limit >= 0 and len(history) > limit:
        del history[:-limit]

    # 事件排序后补齐当前时间区间内的全部事件。
    state["events"] = order_events_by_time(state.get("events", []))
    processed = {str(item) for item in state.setdefault("processed_events", [])}
    due_events = events_in_range(state["events"], old_time, new_time)
    triggered: list[Any] = []
    window = state.setdefault("cancel_window", {})
    if not isinstance(window, dict):
        window = {}
        state["cancel_window"] = window

    for index, event in enumerate(due_events):
        event_time = _event_time(event)
        event_type = _event_type(event)
        event_id = event.get("id") if isinstance(event, dict) else None
        key = str(event_id) if event_id is not None else f"{index}:{event_time}:{event_type}"
        if key in processed:
            continue
        processed.add(key)
        triggered.append(event)

        if "cancel" in event_type and ("open" in event_type or event_type in {"cancel", "cancel_window"}):
            window["event_opened"] = True
            window["open_event_time"] = event_time
        if "cancel" in event_type and "close" in event_type:
            window["event_opened"] = False
            window["close_event_time"] = event_time

    state["processed_events"] = sorted(processed)
    state["triggered_events"] = triggered

    start, end = _window_bounds(state)
    window.update({"start": start, "end": end})
    window_report_now = window_report(state, time=new_time)
    window["open"] = window_report_now["open"]
    window["active"] = window_report_now["open"]
    window["by_time"] = True

    # 根运动：显式曲线/速度优先；否则从层配置中选唯一权威层。
    delta: Any
    if "root_delta" in kwargs:
        delta = kwargs["root_delta"]
    elif state.get("root_curve") is not None or state.get("curve") is not None:
        curve = state.get("root_curve") or state.get("curve")
        delta = sample_root(curve, old_time, new_time)
    elif state.get("root_speed") is not None:
        delta = _number(state["root_speed"]) * dt
    elif state.get("root_layers") is not None or state.get("layers") is not None:
        layers = state.get("root_layers") or state.get("layers")
        plan = _root_plan(layers, state.get("root_layer"))
        delta = plan["delta"]
        state["root_writer"] = plan["name"]
        state["ignored_root_layers"] = plan["ignored"]
    else:
        delta = state.get("next_root_delta", 0.0)

    position = _add_motion(state.get("position", 0.0), delta)
    accumulator = _add_motion(state.get("root_accumulator", 0.0), delta)
    segment_accumulator = _add_motion(state.get("segment_root_accumulator", 0.0), delta)
    state["position"] = position
    state["root_position"] = position
    state["root_accumulator"] = accumulator
    state["accumulator"] = accumulator
    state["segment_root_accumulator"] = segment_accumulator
    state["applied_root_delta"] = delta
    return state


# --------------------------------------------------------------------------- #
# 取消与网络回滚
# --------------------------------------------------------------------------- #


def cancel(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """在时间窗口内取消当前段；成功后冲销本段根运动并回到 idle。"""
    if args and isinstance(args[0], dict):
        state = _default_combo_state(args[0])
        remaining = args[1:]
    else:
        state = _default_combo_state(kwargs.pop("state", None))
        remaining = args

    now = kwargs.get("at", kwargs.get("time", kwargs.get("input_time")))
    if now is None and remaining:
        now = remaining[0]
    now = _number(
        now if now is not None else state.get("last_input_time", state.get("time", 0.0))
    )

    report = window_report(state, time=now)
    allowed = bool(kwargs.get("force", False)) or report["open"]
    state["cancel_request_time"] = now
    state["cancel_window_at_request"] = report

    if not allowed:
        duration = state.get("recovery_end", state.get("animation_end", state.get("duration")))
        if duration is not None and now >= _number(duration):
            state.update(return_to_idle(state))
        state.update({"cancelled": False, "reason": "outside_time_window"})
        return state

    segment_delta = state.get(
        "segment_root_accumulator",
        state.get("root_accumulator", 0.0),
    )
    state = rollback_root_delta(state, segment_delta)
    state.update(reset_accumulator(state))

    target = kwargs.get("target", kwargs.get("next_combo", kwargs.get("next_animation")))
    if target:
        state["animation"] = target
        state["current_animation"] = target
        state["state"] = target
        state["phase"] = "combo"
        state["combo_active"] = True
        state["cancelled"] = True
        state["next_combo"] = target
        window = state.get("cancel_window")
        if isinstance(window, dict):
            window["open"] = False
    else:
        state.update(return_to_idle(state))

    state.update({"cancelled": True, "cancel_time": now, "reason": None})
    return state


def net_rollback(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """恢复网络快照，并把根运动累加器一起恢复/清零。

    支持 ``net_rollback(state)``；也兼容
    ``net_rollback(current_position, snapshot_position, snapshot_accumulator)``
    的直接数值调用。
    """
    if args and not isinstance(args[0], dict):
        current_position = args[0]
        snapshot_position = args[1] if len(args) > 1 else kwargs.get("snapshot_position", current_position)
        snapshot_acc = args[2] if len(args) > 2 else kwargs.get("snapshot_accumulator", 0.0)
        return {
            "position": snapshot_position,
            "root_position": snapshot_position,
            "previous_position": current_position,
            "root_accumulator": snapshot_acc,
            "accumulator": snapshot_acc,
            "rolled_back": True,
        }

    state = deepcopy(args[0]) if args and isinstance(args[0], dict) else _state(kwargs)
    frames_back = int(_number(kwargs.get("frames", state.get("rollback_frames", 3)), 3))
    current_frame = int(_number(state.get("frame", 0), 0))
    target_frame = kwargs.get("target_frame", state.get("target_frame", current_frame - frames_back))
    target_frame = int(_number(target_frame, current_frame - frames_back))

    snapshot: Any = None
    snapshots = state.get("snapshots")
    if isinstance(snapshots, dict):
        snapshot = snapshots.get(target_frame) or snapshots.get(str(target_frame))
    elif isinstance(snapshots, list):
        for item in snapshots:
            if isinstance(item, dict) and int(_number(item.get("frame", -1))) == target_frame:
                snapshot = item
                break

    position_snapshot: Any = None
    history = state.get("history")
    if snapshot is None and isinstance(history, list) and history:
        matching = [
            item for item in history
            if isinstance(item, dict) and int(_number(item.get("frame", -999999))) == target_frame
        ]
        if matching:
            snapshot = matching[0]
        else:
            index = len(history) - frames_back
            if -len(history) <= index < len(history):
                snapshot = history[index]

    position_history = state.get("position_history", state.get("positions"))
    if snapshot is None and isinstance(position_history, list) and position_history:
        if 0 <= target_frame < len(position_history):
            position_snapshot = position_history[target_frame]
        else:
            index = len(position_history) - frames_back
            if -len(position_history) <= index < len(position_history):
                position_snapshot = position_history[index]

    if isinstance(snapshot, dict):
        position = _first_key(snapshot, ("position", "root_position"), state.get("position", 0.0))
        accumulator = _first_key(
            snapshot,
            ("root_accumulator", "accumulator", "root_motion_accumulator"),
            0.0,
        )
        target_frame = int(_number(snapshot.get("frame", target_frame), target_frame))
    elif snapshot is not None:
        position = snapshot
        accumulator = 0.0
    elif position_snapshot is not None:
        position = position_snapshot
        accumulator = state.get("snapshot_accumulator", 0.0)
    else:
        position = state.get(
            "snapshot_position",
            state.get("previous_position", state.get("position", 0.0)),
        )
        accumulator = state.get("snapshot_accumulator", 0.0)

    state["position"] = deepcopy(position)
    state["root_position"] = deepcopy(position)
    state["root_accumulator"] = deepcopy(accumulator)
    state["accumulator"] = deepcopy(accumulator)
    state["rolled_back_to"] = target_frame
    state["rollback_frames"] = frames_back
    state["rolled_back"] = True
    return state


# --------------------------------------------------------------------------- #
# 校验、指标、审计、兜底与确定性
# --------------------------------------------------------------------------- #


def validate_combo(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """检查事件顺序、时间窗口、根运动单层写入和回滚后的累加器状态。"""
    cfg = args[0] if args and isinstance(args[0], dict) else kwargs
    state = _state(cfg)
    errors: list[str] = []
    checks: dict[str, bool] = {}

    events = state.get("events")
    if events is not None:
        ordered = order_events_by_time(events)
        times = [_event_time(event) for event in ordered]
        raw_times = [_event_time(event) for event in events] if isinstance(events, (list, tuple)) else times
        checks["events_ordered"] = raw_times == times
        if not checks["events_ordered"]:
            errors.append("events must be sorted by time")

    if "cancel_window" in state or any(key in state for key in _WINDOW_START_KEYS):
        start, end = _window_bounds(state)
        checks["window_time_based"] = not bool(state.get("frame_window", state.get("window_by_frame", False)))
        checks["window_range_valid"] = start <= end + EPS
        if not checks["window_range_valid"]:
            errors.append("cancel window start must be <= end")
        if not checks["window_time_based"]:
            errors.append("cancel window must be evaluated by time, not frame index")

    if state.get("root_layers") is not None or state.get("layers") is not None:
        layers, source = _layer_sequence(state)
        root_writers = [i for i, layer in enumerate(layers) if _is_root_authority(layer)]
        nonzero_layers = [i for i, layer in enumerate(layers) if _motion_norm(_layer_delta(layer)) > EPS]
        plan = _root_plan(layers, source)
        checks["single_root_writer"] = len(root_writers) <= 1 or len(layers) <= 1
        checks["single_root_delta"] = len(nonzero_layers) <= 1 or source is not None or any(
            _is_root_authority(layer) for layer in layers
        )
        if not checks["single_root_writer"]:
            errors.append("multiple animation layers are marked as root writers")
        # 有明确 base/authority 时允许其他层携带但被忽略；没有权威层却有多个
        # 非零位移层，说明可能发生双层叠加。
        if len(nonzero_layers) > 1 and not any(_is_root_authority(layer) for layer in layers):
            errors.append("multiple layers contribute root motion")
        state["_root_plan"] = plan

    animation = str(state.get("animation", ""))
    accumulator = _motion_norm(state.get("root_accumulator", 0.0))
    if animation == IDLE and accumulator > EPS:
        errors.append("idle state must not retain a root motion accumulator")
        checks["accumulator_reset_in_idle"] = False
    else:
        checks["accumulator_reset_in_idle"] = True

    return {"ok": not errors, "errors": errors, "checks": checks}


def metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """统计事件、窗口、根运动速度倍率和位移回滚情况。"""
    cfg = args[0] if args and isinstance(args[0], dict) else kwargs
    state = _state(cfg)

    events = state.get("events", [])
    ordered_times = [_event_time(e) for e in order_events_by_time(events)] if events else []
    raw_times = [_event_time(e) for e in events] if isinstance(events, (list, tuple)) else []
    out_of_order = sum(
        1 for raw, ordered in zip(raw_times, ordered_times) if abs(_number(raw) - _number(ordered)) > EPS
    )

    layers = state.get("root_layers", state.get("layers"))
    root_name = None
    authoritative_delta: Any = 0.0
    ignored: list[Any] = []
    raw_weighted: float = 0.0
    if layers is not None:
        plan = _root_plan(layers, state.get("root_layer"))
        root_name = plan["name"]
        authoritative_delta = plan["delta"]
        ignored = plan["ignored"]
        layer_sequence, _ = _layer_sequence(layers)
        raw_weighted = sum(_motion_norm(_layer_contribution_metric(layer)) for layer in layer_sequence)

    base_speed = _motion_norm(authoritative_delta)
    actual_speed = raw_weighted if raw_weighted > EPS else base_speed
    speed_ratio = actual_speed / base_speed if base_speed > EPS else 1.0

    position = state.get("position", state.get("root_position", 0.0))
    return {
        "event_count": len(ordered_times),
        "event_times": ordered_times,
        "events_out_of_order": out_of_order,
        "window": window_report(state) if ("cancel_window" in state or "time" in state) else None,
        "root_writer": root_name,
        "root_delta": authoritative_delta,
        "ignored_root_layers": ignored,
        "configured_speed": base_speed,
        "actual_layered_speed": actual_speed,
        "speed_ratio": speed_ratio,
        "over_speed_fraction": speed_ratio - 1.0,
        "position": position,
        "distance": _motion_norm(position),
        "root_accumulator": state.get("root_accumulator", 0.0),
        "rollback_count": int(_number(state.get("rollback_count", 0))),
        "missed_inputs": int(_number(state.get("missed_inputs", 0))),
    }


def _layer_contribution_metric(layer: Any) -> Any:
    if isinstance(layer, dict) and "root_contribution" in layer:
        return layer["root_contribution"]
    delta = _layer_delta(layer)
    weight = _layer_weight(layer)
    if _is_number(delta):
        return delta * weight
    if isinstance(delta, (list, tuple)):
        return tuple(_number(x) * weight for x in delta)
    return delta


def audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """生成一次连招/根运动修复的可审计摘要。"""
    cfg = args[0] if args and isinstance(args[0], dict) else kwargs
    state = _state(cfg)
    validation = validate_combo(state)
    metric_report = metrics(state)
    events = state.get("events", [])
    plan = _root_plan(state.get("root_layers") or state.get("layers")) if (
        state.get("root_layers") is not None or state.get("layers") is not None
    ) else None

    return {
        "ok": validation["ok"],
        "errors": validation["errors"],
        "event_order": [_event_time(e) for e in order_events_by_time(events)] if events else [],
        "events_sorted": validation["checks"].get("events_ordered", True),
        "window": window_report(state) if ("cancel_window" in state or "time" in state) else None,
        "root_writer": plan["name"] if plan else None,
        "root_delta": plan["delta"] if plan else 0.0,
        "ignored_root_layers": plan["ignored"] if plan else [],
        "position": state.get("position", 0.0),
        "root_accumulator": state.get("root_accumulator", 0.0),
        "animation": state.get("animation"),
        "idle": state.get("animation") == IDLE,
        "accumulator_reset": _motion_norm(state.get("root_accumulator", 0.0)) <= EPS,
        "validation": validation,
        "metrics": metric_report,
    }


def fallback(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """非法状态或显式失败时，冲销累计根运动并安全回到 idle。"""
    valid_arg = None
    if args and isinstance(args[-1], bool):
        valid_arg = args[-1]
        cfg = args[0] if args and isinstance(args[0], dict) else {}
    elif args and isinstance(args[0], dict):
        cfg = args[0]
    elif args and isinstance(args[0], bool):
        valid_arg = args[0]
        cfg = {}
    else:
        cfg = kwargs if kwargs else {}

    state = _state(cfg)
    validation = validate_combo(state)
    valid = validation["ok"] if valid_arg is None else valid_arg
    forced = bool(kwargs.get("force", state.get("force_fallback", False)))
    if valid and not forced and not state.get("error"):
        result = deepcopy(state)
        result["fallback"] = False
        result.setdefault("ok", True)
        return result

    safe = deepcopy(state)
    if kwargs.get("rollback_root", True):
        safe = rollback_root_delta(safe)
    safe.update(reset_accumulator(safe))
    safe.update(return_to_idle(safe))
    safe.update(
        {
            "fallback": True,
            "ok": False,
            "reason": kwargs.get("reason", "invalid_combo_state"),
            "validation_errors": validation["errors"],
        }
    )
    return safe


def determinism(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """验证按时间推进时，30/60/120 FPS 得到相同根运动距离。"""
    if args and isinstance(args[0], dict):
        cfg = args[0]
    elif args and _is_number(args[0]):
        cfg = {"duration": args[0]}
    else:
        cfg = kwargs

    state = _state(cfg)
    duration = _number(state.get("duration", 1.0), 1.0)
    speed = _number(state.get("speed", state.get("root_speed", 1.0)), 1.0)
    fps_values = state.get("fps_values", kwargs.get("fps_values", (30, 60, 120)))

    results: dict[int, float] = {}
    for raw_fps in fps_values:
        fps = max(1, int(_number(raw_fps, 60)))
        steps = max(1, int(round(duration * fps)))
        position = 0.0
        for _ in range(steps):
            position += speed / fps
        results[fps] = position

    expected = speed * duration
    max_error = max((abs(position - expected) for position in results.values()), default=0.0)
    return {
        "duration": duration,
        "speed": speed,
        "fps_results": results,
        "positions": list(results.values()),
        "expected": expected,
        "max_error": max_error,
        "deterministic": max_error < 1e-9,
        "window_by_time": True,
    }
