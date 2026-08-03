#!/usr/bin/env python3
"""Generate responsive, claim-bounded verl bridge diagrams."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

from miniverl.bridge.contract import VERL_COMMIT, VERL_TAG

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SHORT_COMMIT = VERL_COMMIT[:8]


def _shell(*, width: int, height: int, title: str, description: str, body: list[str]) -> str:
    style = (
        "text{font-family:'DejaVu Sans','Segoe UI',sans-serif;fill:#edf4ff}"
        ".title{font-size:30px;font-weight:760}.sub{font-size:17px;fill:#aebbd2}"
        ".layer{font-size:20px;font-weight:760}.body{font-size:16px;fill:#c2cee1}"
        ".role{font-size:15px;font-weight:700}.status{font-size:22px;font-weight:800}"
        ".foot{font-size:15px;fill:#9fb0ca}"
    )
    return "".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            f'<rect width="{width}" height="{height}" rx="22" fill="#060a14"/>',
            f"<style>{style}</style>",
            *body,
            "</svg>\n",
        ]
    )


def _desktop() -> str:
    title = "Scale-out bridge: verified artifacts, bounded claims"
    description = (
        "Three verified layers connect miniVERL local runtime artifacts to a pinned verl "
        "parse and load smoke. A dashed arrow leads to distributed execution marked not tested."
    )
    body = [
        f'<text class="title" data-role="diagram-label" x="48" y="58">{title}</text>',
        f'<text class="sub" data-role="diagram-label" x="48" y="88">pinned upstream {VERL_TAG} · {SHORT_COMMIT} · independent project; no endorsement</text>',
        '<rect x="70" y="122" width="980" height="132" rx="18" fill="#0d1d34" stroke="#2f6fa2" stroke-width="2"/>',
        '<text class="layer" data-role="diagram-label" x="98" y="154">1 · miniVERL local runtime</text>',
        '<text class="body" data-role="diagram-label" x="98" y="181">single-GPU training, evaluation and portable provenance</text>',
        '<rect x="98" y="199" width="210" height="38" rx="9" fill="#12304d"/>',
        '<text class="role" data-role="diagram-label" x="116" y="224">teacher role · targets</text>',
        '<rect x="322" y="199" width="214" height="38" rx="9" fill="#2d2343"/>',
        '<text class="role" data-role="diagram-label" x="340" y="224">reference role · DPO</text>',
        '<rect x="550" y="199" width="224" height="38" rx="9" fill="#16372f"/>',
        '<text class="role" data-role="diagram-label" x="568" y="224">reward role · verifier</text>',
        '<rect x="788" y="199" width="234" height="38" rx="9" fill="#352d17"/>',
        '<text class="role" data-role="diagram-label" x="806" y="224">student · local updates</text>',
        '<line x1="560" y1="254" x2="560" y2="285" stroke="#80c7ff" stroke-width="4"/>',
        '<polygon points="551,278 569,278 560,291" fill="#80c7ff"/>',
        '<rect x="100" y="291" width="920" height="148" rx="18" fill="#101f38" stroke="#4e87b7" stroke-width="2"/>',
        '<text class="layer" data-role="diagram-label" x="128" y="326">2 · portable artifact bundle</text>',
        '<text class="body" data-role="diagram-label" x="128" y="353">standard formats with explicit config and provenance boundaries</text>',
        '<rect x="128" y="374" width="132" height="40" rx="10" fill="#16314f"/><text class="role" data-role="diagram-label" x="153" y="400">PEFT</text>',
        '<rect x="274" y="374" width="144" height="40" rx="10" fill="#16314f"/><text class="role" data-role="diagram-label" x="293" y="400">safetensors</text>',
        '<rect x="432" y="374" width="132" height="40" rx="10" fill="#16314f"/><text class="role" data-role="diagram-label" x="459" y="400">Parquet</text>',
        '<rect x="578" y="374" width="174" height="40" rx="10" fill="#16314f"/><text class="role" data-role="diagram-label" x="596" y="400">resolved config</text>',
        '<rect x="766" y="374" width="226" height="40" rx="10" fill="#16314f"/><text class="role" data-role="diagram-label" x="788" y="400">typed provenance</text>',
        '<line x1="560" y1="439" x2="560" y2="470" stroke="#80c7ff" stroke-width="4"/>',
        '<polygon points="551,463 569,463 560,476" fill="#80c7ff"/>',
        '<rect x="100" y="476" width="920" height="108" rx="18" fill="#10283b" stroke="#3c9f9a" stroke-width="2"/>',
        '<text class="layer" data-role="diagram-label" x="128" y="515">3 · pinned upstream parse/load smoke</text>',
        f'<text class="body" data-role="diagram-label" x="128" y="545">verl {VERL_TAG} at {SHORT_COMMIT} · config parse + PEFT/safetensors/Parquet structural load checks</text>',
        '<text class="body" data-role="diagram-label" x="128" y="568">verified boundary: artifact interchange and the documented profile subset</text>',
        '<line x1="560" y1="584" x2="560" y2="621" stroke="#f29d65" stroke-width="4" stroke-dasharray="10 8"/>',
        '<polygon points="551,614 569,614 560,627" fill="#f29d65"/>',
        '<rect x="100" y="627" width="920" height="112" rx="18" fill="#321b20" stroke="#ef835f" stroke-width="3"/>',
        '<text class="status" data-role="diagram-label" x="560" y="671" text-anchor="middle">Distributed execution: NOT TESTED</text>',
        '<text class="body" data-role="diagram-label" x="560" y="701" text-anchor="middle">No Ray / FSDP / vLLM job ran · no OPD-to-PPO semantic-parity claim</text>',
        '<text class="foot" data-role="diagram-label" x="560" y="726" text-anchor="middle">unverified execution layer</text>',
    ]
    return _shell(width=1120, height=760, title=title, description=description, body=body)


def _mobile() -> str:
    title = "miniVERL → verl bridge"
    description = (
        "Vertical mobile diagram with local runtime, portable bundle and pinned upstream smoke "
        "as verified layers, followed by Distributed execution: NOT TESTED."
    )
    body = [
        f'<text class="title" data-role="diagram-label" x="20" y="46" style="font-size:24px">{title}</text>',
        f'<text class="sub" data-role="diagram-label" x="20" y="75" style="font-size:15px">{VERL_TAG} · {SHORT_COMMIT}</text>',
        '<text class="sub" data-role="diagram-label" x="20" y="98" style="font-size:14px">independent project; no endorsement</text>',
        '<rect x="20" y="122" width="350" height="242" rx="16" fill="#0d1d34" stroke="#2f6fa2" stroke-width="2"/>',
        '<text class="layer" data-role="diagram-label" x="40" y="154">1 · miniVERL local runtime</text>',
        '<text class="body" data-role="diagram-label" x="40" y="181" style="font-size:14px">single-GPU training + evaluation</text>',
        '<rect x="40" y="199" width="310" height="32" rx="8" fill="#12304d"/><text class="role" data-role="diagram-label" x="55" y="220">teacher role · targets</text>',
        '<rect x="40" y="239" width="310" height="32" rx="8" fill="#2d2343"/><text class="role" data-role="diagram-label" x="55" y="260">reference role · DPO</text>',
        '<rect x="40" y="279" width="310" height="32" rx="8" fill="#16372f"/><text class="role" data-role="diagram-label" x="55" y="300">reward role · verifier</text>',
        '<rect x="40" y="319" width="310" height="32" rx="8" fill="#352d17"/><text class="role" data-role="diagram-label" x="55" y="340">student · local updates</text>',
        '<line x1="195" y1="364" x2="195" y2="393" stroke="#80c7ff" stroke-width="4"/><polygon points="186,386 204,386 195,399" fill="#80c7ff"/>',
        '<rect x="20" y="399" width="350" height="226" rx="16" fill="#101f38" stroke="#4e87b7" stroke-width="2"/>',
        '<text class="layer" data-role="diagram-label" x="40" y="433">2 · portable artifact bundle</text>',
        '<text class="body" data-role="diagram-label" x="40" y="460" style="font-size:14px">standard, reviewable interchange</text>',
        '<rect x="40" y="482" width="145" height="40" rx="9" fill="#16314f"/><text class="role" data-role="diagram-label" x="75" y="507">PEFT</text>',
        '<rect x="205" y="482" width="145" height="40" rx="9" fill="#16314f"/><text class="role" data-role="diagram-label" x="226" y="507">safetensors</text>',
        '<rect x="40" y="536" width="145" height="40" rx="9" fill="#16314f"/><text class="role" data-role="diagram-label" x="75" y="561">Parquet</text>',
        '<rect x="205" y="536" width="145" height="40" rx="9" fill="#16314f"/><text class="role" data-role="diagram-label" x="225" y="561">config</text>',
        '<rect x="40" y="590" width="310" height="24" rx="7" fill="#16314f"/><text class="role" data-role="diagram-label" x="100" y="608" style="font-size:13px">typed provenance</text>',
        '<line x1="195" y1="625" x2="195" y2="654" stroke="#80c7ff" stroke-width="4"/><polygon points="186,647 204,647 195,660" fill="#80c7ff"/>',
        '<rect x="20" y="660" width="350" height="154" rx="16" fill="#10283b" stroke="#3c9f9a" stroke-width="2"/>',
        '<text class="layer" data-role="diagram-label" x="40" y="696">3 · pinned upstream smoke</text>',
        f'<text class="body" data-role="diagram-label" x="40" y="725" style="font-size:14px">verl {VERL_TAG} at {SHORT_COMMIT}</text>',
        '<text class="body" data-role="diagram-label" x="40" y="751" style="font-size:14px">config parse + artifact load checks</text>',
        '<text class="body" data-role="diagram-label" x="40" y="781" style="font-size:13px">verified: documented profile subset</text>',
        '<line x1="195" y1="814" x2="195" y2="847" stroke="#f29d65" stroke-width="4" stroke-dasharray="10 8"/><polygon points="186,840 204,840 195,853" fill="#f29d65"/>',
        '<rect x="20" y="853" width="350" height="174" rx="16" fill="#321b20" stroke="#ef835f" stroke-width="3"/>',
        '<text class="status" data-role="diagram-label" x="195" y="892" text-anchor="middle" style="font-size:19px">Distributed execution:</text>',
        '<text class="status" data-role="diagram-label" x="195" y="935" text-anchor="middle" style="font-size:24px">NOT TESTED</text>',
        '<text class="body" data-role="diagram-label" x="195" y="968" text-anchor="middle" style="font-size:13px">No Ray / FSDP / vLLM job ran</text>',
        '<text class="body" data-role="diagram-label" x="195" y="992" text-anchor="middle" style="font-size:13px">No OPD-to-PPO semantic parity</text>',
        '<text class="foot" data-role="diagram-label" x="195" y="1017" text-anchor="middle" style="font-size:13px">unverified execution layer</text>',
    ]
    return _shell(width=390, height=1048, title=title, description=description, body=body)


def render_diagrams() -> dict[str, str]:
    return {
        "verl-bridge-architecture.svg": _desktop(),
        "verl-bridge-architecture-mobile.svg": _mobile(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, content in render_diagrams().items():
        path = DOCS / name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"generated bridge diagram is stale: {path}")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
