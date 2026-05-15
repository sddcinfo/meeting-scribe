"""Canonical button-feedback label catalog + resolver.

When a speakerphone button is pressed, the daemon dispatches the action
*and* hits ``POST /api/internal/speakerphone/speak`` with a stable
``label_id``. The server side resolves that label_id to the spoken text
in the configured language. The catalog below is the single source of
truth for the text — translations are hard-coded into every TTS-native
language registered in :mod:`meeting_scribe.languages`.

Why hard-coded translations: the strings are short, the set is small
(~14 labels × 10 languages = ~140 strings), and accuracy matters more
than scale. Doing on-the-fly translation through the 35B model would
add latency and risk mistranslations on simple phrases like "Mic
muted". Custom override paths exist for user-specific phrasing.

Schema:

* ``LABELS[label_id][lang_code]`` → spoken text.
* ``CANONICAL_LABELS`` — frozenset of all label_ids (used by mapping
  validation to reject typos in the user's override config).
* ``resolve(label_id, language, overrides)`` — precedence:
    1. ``overrides[label_id][language]`` (operator's saved override
       OR an inline override passed by the preview route)
    2. ``LABELS[label_id][language]`` (catalog entry for the language)
    3. ``LABELS[label_id]["en"]`` (English fallback — always present)
  Unknown ``label_id`` raises :class:`LabelNotFoundError` so the
  caller can return 400 instead of synthesizing garbage.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final


class LabelNotFoundError(KeyError):
    """Raised when ``resolve`` is called with an unknown label_id.

    Subclass of ``KeyError`` so existing ``except KeyError`` blocks keep
    working; using a dedicated class lets the router emit a more
    specific 400 message ("unknown label_id: 'foo'") without catching
    every other KeyError.
    """


# Canonical translations. Every label_id must have an entry for every
# TTS-native language code in ``languages.LANGUAGE_REGISTRY``. The
# coverage test in ``tests/test_speakerphone_labels.py`` enforces this
# at CI time so a new language doesn't silently miss labels.
#
# Translation principles applied:
#   * Short — feedback should not interrupt or feel chatty. Most
#     labels are 2–4 words.
#   * Idiomatic — "Mic on" rather than "Mic unmuted" reads more
#     natural; "出力は英語のみ" reads naturally in Japanese.
#   * Action-state consistent — "Mic muted" / "Mic on", "Interpretation
#     on" / "Interpretation off" — verb-state forms match across pairs.
LABELS: Final[dict[str, dict[str, str]]] = {
    # ── Stateless directional system labels ──────────────────────────
    "volume_up": {
        "en": "Volume up",
        "zh": "音量增大",
        "ja": "音量アップ",
        "ko": "볼륨 업",
        "fr": "Volume plus",
        "de": "Lauter",
        "es": "Volumen arriba",
        "it": "Volume su",
        "pt": "Volume mais",
        "ru": "Громче",
    },
    "volume_down": {
        "en": "Volume down",
        "zh": "音量减小",
        "ja": "音量ダウン",
        "ko": "볼륨 다운",
        "fr": "Volume moins",
        "de": "Leiser",
        "es": "Volumen abajo",
        "it": "Volume giù",
        "pt": "Volume menos",
        "ru": "Тише",
    },
    # Note: the system Vol+/Vol-/Mute keys are SILENT by design — the
    # audio change is its own feedback (2026-05-13). The ``system_*``
    # speech-catalog entries were removed when the consumer-page
    # announcements were dropped. ``system_mute_toggled`` survives as
    # an *action sentinel* in ``KERNEL_KEY_TO_CONSUMER_LABEL`` —
    # ``_apply_consumer_action`` matches on it to dispatch ``wpctl
    # set-mute toggle`` — but no longer has a speech translation.
    # ── Mic mute (toggle, state-resolved by daemon) ──────────────────
    "mic_muted": {
        "en": "Mic muted",
        "zh": "麦克风静音",
        "ja": "マイクミュート",
        "ko": "마이크 음소거",
        "fr": "Micro coupé",
        "de": "Mikro stumm",
        "es": "Micro silenciado",
        "it": "Microfono muto",
        "pt": "Microfone mudo",
        "ru": "Микрофон выключен",
    },
    "mic_unmuted": {
        "en": "Mic on",
        "zh": "麦克风已开",
        "ja": "マイク オン",
        "ko": "마이크 켜짐",
        "fr": "Micro activé",
        "de": "Mikro an",
        "es": "Micro activado",
        "it": "Microfono attivo",
        "pt": "Microfone ligado",
        "ru": "Микрофон включён",
    },
    # ── TTS direction cycle (Phone-button short press) ───────────────
    "tts_dir_en": {
        "en": "Output English only",
        "zh": "仅输出英语",
        "ja": "出力は英語のみ",
        "ko": "영어만 출력",
        "fr": "Sortie anglais uniquement",
        "de": "Ausgabe nur Englisch",
        "es": "Salida solo en inglés",
        "it": "Solo uscita inglese",
        "pt": "Saída apenas em inglês",
        "ru": "Только английский на выходе",
    },
    "tts_dir_ja": {
        "en": "Output Japanese only",
        "zh": "仅输出日语",
        "ja": "出力は日本語のみ",
        "ko": "일본어만 출력",
        "fr": "Sortie japonais uniquement",
        "de": "Ausgabe nur Japanisch",
        "es": "Salida solo en japonés",
        "it": "Solo uscita giapponese",
        "pt": "Saída apenas em japonês",
        "ru": "Только японский на выходе",
    },
    "tts_dir_zh": {
        "en": "Output Chinese only",
        "zh": "仅输出中文",
        "ja": "出力は中国語のみ",
        "ko": "중국어만 출력",
        "fr": "Sortie chinois uniquement",
        "de": "Ausgabe nur Chinesisch",
        "es": "Salida solo en chino",
        "it": "Solo uscita cinese",
        "pt": "Saída apenas em chinês",
        "ru": "Только китайский на выходе",
    },
    "tts_dir_ko": {
        "en": "Output Korean only",
        "zh": "仅输出韩语",
        "ja": "出力は韓国語のみ",
        "ko": "한국어만 출력",
        "fr": "Sortie coréen uniquement",
        "de": "Ausgabe nur Koreanisch",
        "es": "Salida solo en coreano",
        "it": "Solo uscita coreano",
        "pt": "Saída apenas em coreano",
        "ru": "Только корейский на выходе",
    },
    "tts_dir_fr": {
        "en": "Output French only",
        "zh": "仅输出法语",
        "ja": "出力はフランス語のみ",
        "ko": "프랑스어만 출력",
        "fr": "Sortie français uniquement",
        "de": "Ausgabe nur Französisch",
        "es": "Salida solo en francés",
        "it": "Solo uscita francese",
        "pt": "Saída apenas em francês",
        "ru": "Только французский на выходе",
    },
    "tts_dir_de": {
        "en": "Output German only",
        "zh": "仅输出德语",
        "ja": "出力はドイツ語のみ",
        "ko": "독일어만 출력",
        "fr": "Sortie allemand uniquement",
        "de": "Ausgabe nur Deutsch",
        "es": "Salida solo en alemán",
        "it": "Solo uscita tedesco",
        "pt": "Saída apenas em alemão",
        "ru": "Только немецкий на выходе",
    },
    "tts_dir_es": {
        "en": "Output Spanish only",
        "zh": "仅输出西班牙语",
        "ja": "出力はスペイン語のみ",
        "ko": "스페인어만 출력",
        "fr": "Sortie espagnol uniquement",
        "de": "Ausgabe nur Spanisch",
        "es": "Salida solo en español",
        "it": "Solo uscita spagnolo",
        "pt": "Saída apenas em espanhol",
        "ru": "Только испанский на выходе",
    },
    "tts_dir_it": {
        "en": "Output Italian only",
        "zh": "仅输出意大利语",
        "ja": "出力はイタリア語のみ",
        "ko": "이탈리아어만 출력",
        "fr": "Sortie italien uniquement",
        "de": "Ausgabe nur Italienisch",
        "es": "Salida solo en italiano",
        "it": "Solo uscita italiano",
        "pt": "Saída apenas em italiano",
        "ru": "Только итальянский на выходе",
    },
    "tts_dir_pt": {
        "en": "Output Portuguese only",
        "zh": "仅输出葡萄牙语",
        "ja": "出力はポルトガル語のみ",
        "ko": "포르투갈어만 출력",
        "fr": "Sortie portugais uniquement",
        "de": "Ausgabe nur Portugiesisch",
        "es": "Salida solo en portugués",
        "it": "Solo uscita portoghese",
        "pt": "Saída apenas em português",
        "ru": "Только португальский на выходе",
    },
    "tts_dir_ru": {
        "en": "Output Russian only",
        "zh": "仅输出俄语",
        "ja": "出力はロシア語のみ",
        "ko": "러시아어만 출력",
        "fr": "Sortie russe uniquement",
        "de": "Ausgabe nur Russisch",
        "es": "Salida solo en ruso",
        "it": "Solo uscita russo",
        "pt": "Saída apenas em russo",
        "ru": "Только русский на выходе",
    },
    "tts_dir_all": {
        "en": "Output both languages",
        "zh": "输出两种语言",
        "ja": "両方の言語を出力",
        "ko": "두 언어 모두 출력",
        "fr": "Sortie dans les deux langues",
        "de": "Ausgabe in beiden Sprachen",
        "es": "Salida en ambos idiomas",
        "it": "Uscita in entrambe le lingue",
        "pt": "Saída em ambos os idiomas",
        "ru": "Оба языка на выходе",  # noqa: RUF001 — Cyrillic О is correct for Russian
    },
    # ── Interpretation toggle (Phone-button long press) ──────────────
    "interp_on": {
        "en": "Interpretation on",
        "zh": "口译开启",
        "ja": "通訳オン",
        "ko": "통역 켜짐",
        "fr": "Interprétation activée",
        "de": "Verdolmetschung an",
        "es": "Interpretación activada",
        "it": "Interpretazione attiva",
        "pt": "Interpretação ligada",
        "ru": "Перевод включён",
    },
    "interp_off": {
        "en": "Interpretation off",
        "zh": "口译关闭",
        "ja": "通訳オフ",
        "ko": "통역 꺼짐",
        "fr": "Interprétation désactivée",
        "de": "Verdolmetschung aus",
        "es": "Interpretación desactivada",
        "it": "Interpretazione inattiva",
        "pt": "Interpretação desligada",
        "ru": "Перевод выключен",
    },
    # ── Meeting record toggle (Teams button) ─────────────────────────
    "meeting_started": {
        "en": "Recording started",
        "zh": "录音已开始",
        "ja": "録音開始",
        "ko": "녹음 시작",
        "fr": "Enregistrement démarré",
        "de": "Aufnahme gestartet",
        "es": "Grabación iniciada",
        "it": "Registrazione avviata",
        "pt": "Gravação iniciada",
        "ru": "Запись начата",
    },
    "meeting_stopped": {
        "en": "Recording stopped",
        "zh": "录音已停止",
        "ja": "録音停止",
        "ko": "녹음 중지",
        "fr": "Enregistrement arrêté",
        "de": "Aufnahme beendet",
        "es": "Grabación detenida",
        "it": "Registrazione fermata",
        "pt": "Gravação parada",
        "ru": "Запись остановлена",
    },
}

CANONICAL_LABELS: Final[frozenset[str]] = frozenset(LABELS)


def resolve(
    label_id: str,
    language: str,
    overrides: Mapping[str, Mapping[str, str]] | None = None,
) -> str:
    """Resolve a label_id + language to spoken text.

    Precedence (first non-empty wins):
      1. ``overrides[label_id][language]`` — operator's per-(label,lang)
         string (saved in the mapping doc OR sent inline by the
         preview route).
      2. ``LABELS[label_id][language]`` — built-in translation.
      3. ``LABELS[label_id]["en"]`` — English fallback (always present
         in the catalog since the coverage test enforces ``en`` for
         every label_id).

    Raises :class:`LabelNotFoundError` if ``label_id`` isn't in the
    catalog. The text resolution never raises for missing language —
    the English fallback catches that case.
    """
    if label_id not in LABELS:
        raise LabelNotFoundError(f"unknown label_id: {label_id!r}")

    if overrides:
        per_label = overrides.get(label_id)
        if isinstance(per_label, Mapping):
            override_text = per_label.get(language)
            if isinstance(override_text, str) and override_text.strip():
                return override_text

    catalog_entry = LABELS[label_id]
    text = catalog_entry.get(language)
    if isinstance(text, str) and text.strip():
        return text
    # English is always present (coverage test enforces this).
    return catalog_entry["en"]


__all__ = ["CANONICAL_LABELS", "LABELS", "LabelNotFoundError", "resolve"]
