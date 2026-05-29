from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable


COMMENT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "speak": {"type": "boolean"},
        "priority": {"type": "number", "minimum": 0, "maximum": 1},
        "line": {"type": "string"},
        "reason": {"type": "string"},
        "cooldown_sec": {"type": "integer", "minimum": 0, "maximum": 120},
    },
    "required": ["speak", "priority", "line", "reason", "cooldown_sec"],
}


@dataclass(frozen=True)
class ImagePayload:
    label: str
    data: bytes
    mime_type: str = "image/jpeg"

    def to_part(self) -> dict[str, object]:
        return {
            "inlineData": {
                "mimeType": self.mime_type,
                "data": base64.b64encode(self.data).decode("ascii"),
            }
        }


class Commenter:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        disabled: bool = False,
        timeout_sec: float = 20.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        self.model = model if model is not None else os.environ.get("GEMINI_MODEL")
        self.disabled = disabled
        self.timeout_sec = float(timeout_sec)

    @property
    def gemini_enabled(self) -> bool:
        return not self.disabled and bool(self.api_key) and bool(self.model)

    def generate(
        self,
        request: dict[str, object],
        images: Iterable[ImagePayload] = (),
    ) -> dict[str, object]:
        fallback = self.template_comment(request)
        if not self.gemini_enabled:
            return fallback

        try:
            response = self._generate_with_gemini(request, list(images))
            return self._normalize_response(response, source="gemini", fallback=fallback)
        except Exception as exc:  # Network/API errors should not break video analysis.
            fallback["source"] = "template_fallback"
            fallback["reason"] = f"{fallback.get('reason', '')}; gemini_error={exc.__class__.__name__}"
            return fallback

    def _generate_with_gemini(self, request: dict[str, object], images: list[ImagePayload]) -> dict[str, object]:
        assert self.api_key is not None
        assert self.model is not None
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        parts: list[dict[str, object]] = [
            {
                "text": (
                    "あなたはSplatoon 3のプレイヤー本人にだけ辛口で短く突っ込む実況補助です。\n"
                    "人格、属性、他プレイヤー名、実在個人への攻撃は禁止。事実と推定を混ぜず、"
                    "低confidenceの内容は断定しないでください。\n"
                    "出力は指定JSONのみ。lineは日本語で12から45文字程度、1文にしてください。\n"
                    "ガチヤグラでは objective.tower_owner, tower_progress, screen_position を根拠にして、"
                    "ヤグラが進んでいる時の死亡を優先して指摘してください。\n"
                    "死亡時画像がある場合、順番は t-2s, t-1s, death_frame, death_roi です。\n"
                    "状態JSON:\n"
                    f"{json.dumps(request, ensure_ascii=False, separators=(',', ':'))}"
                )
            }
        ]
        for image in images:
            parts.append({"text": f"image_label={image.label}"})
            parts.append(image.to_part())

        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.75,
                "maxOutputTokens": 256,
                "responseMimeType": "application/json",
                "responseJsonSchema": COMMENT_SCHEMA,
            },
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc

        text = self._extract_text(payload)
        return json.loads(text)

    @staticmethod
    def _extract_text(payload: dict[str, object]) -> str:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("Gemini response has no candidates")
        content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list) or not parts:
            raise RuntimeError("Gemini response has no text parts")
        texts = [str(part.get("text", "")) for part in parts if isinstance(part, dict) and part.get("text")]
        if not texts:
            raise RuntimeError("Gemini response text is empty")
        return "".join(texts)

    @staticmethod
    def _normalize_response(
        response: dict[str, object],
        *,
        source: str,
        fallback: dict[str, object],
    ) -> dict[str, object]:
        line = str(response.get("line", "")).strip()
        if not line:
            return fallback
        result = {
            "speak": bool(response.get("speak", True)),
            "priority": _clamp_float(response.get("priority", 0.7), 0.0, 1.0),
            "line": line[:80],
            "reason": str(response.get("reason", ""))[:300],
            "cooldown_sec": int(_clamp_float(response.get("cooldown_sec", 8), 0, 120)),
            "source": source,
        }
        return result

    @staticmethod
    def template_comment(request: dict[str, object]) -> dict[str, object]:
        facts = request.get("facts")
        facts_list = [str(item) for item in facts] if isinstance(facts, list) else []
        inferences = request.get("inferences")
        inference_names = []
        if isinstance(inferences, list):
            for item in inferences:
                if isinstance(item, dict):
                    inference_names.append(str(item.get("type", "")))

        if "death_with_special_ready" in facts_list:
            line = "スペシャル抱えて死ぬな。飾りじゃないぞ。"
            reason = "player died while special was ready"
            priority = 0.95
        elif "died_while_enemy_tower_pushing" in inference_names:
            line = "ヤグラ進んでるのに死ぬな。まず止めろ。"
            reason = "player died while enemy tower was pushing"
            priority = 0.92
        elif "late_game_death" in facts_list:
            line = "終盤で死ぬな。今のデスが一番重い。"
            reason = "player died in late game"
            priority = 0.9
        elif "outnumbered_death" in inference_names:
            line = "人数不利で突っ込むな。今のは献上。"
            reason = "player probably pushed while outnumbered"
            priority = 0.88
        else:
            line = "また落ちた。今の判断、だいぶ雑。"
            reason = "generic player death"
            priority = 0.75
        return {
            "speak": True,
            "priority": priority,
            "line": line,
            "reason": reason,
            "cooldown_sec": 8,
            "source": "template_fallback",
        }


def _clamp_float(value: object, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))
