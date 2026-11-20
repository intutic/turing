import time
import pytest
from turing.serving.engine import RequestRejected
from turing.serving.ollama_api import render_go_template

V = {"System": "sys", "Prompt": "hi", "Response": ""}


@pytest.mark.parametrize("template, expected", [
    ("{{ .Prompt }}", "hi"),
    ("{{- .System -}}|{{.Prompt}}", "sys|hi"),
    ("{{ if .System }}<s>{{ .System }}</s>{{ end }}{{ .Prompt }}", "<s>sys</s>hi"),
    ("{{ if .Response }}R:{{ .Response }}{{ end }}[{{ .Prompt }}]", "[hi]"),
    ("{{- if .Missing -}}x{{- end -}}y", "y"),
    ("plain text", "plain text"),
])
def test_go_template_renders(template, expected):
    assert render_go_template(template, V) == expected


@pytest.mark.parametrize("template", ["{{ .Messages }}", "{{ range .X }}{{ end }}", "{{ if .System }}open",
                                      "{{ end }}", "{{ .Prompt | upper }}"])
def test_go_template_refuses_what_it_cannot_render(template):
    with pytest.raises(RequestRejected) as e:
        render_go_template(template, V)
    assert e.value.status_code == 400


def test_go_template_keeps_braces_in_values():
    assert render_go_template("[{{ .Prompt }}]", {"Prompt": "f({{x}})"}) == "[f({{x}})]"


def test_go_template_is_linear_on_adversarial_input():
    # the CodeQL ReDoS pattern: many unclosed "{{if .a}}" openers
    t = "{{if .a}}" * 20_000 + "a" * 20_000
    start = time.perf_counter()
    with pytest.raises(RequestRejected):
        render_go_template(t, {"a": "1"})
    assert time.perf_counter() - start < 1.0
