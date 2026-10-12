# Moonwalker hallucination test — Gemma 4 12B vs Gemma 4 31B vs Qwen3.8-27B

**Prompt (verbatim):** *"without using internet, prepare a list of all people who ever walked on the moon surface, provide 1-2 sentence background information on each"*

**Setup:**
- Hardware: M5 Pro, 48 GB, llama.cpp 0.6.0, through `localllm-serve`.
- Sampling: temperature 0.7, top_p 0.95, one sample per run.
- Builds: 12B `q5_k`, 31B `qat_q4_0`, Qwen `ud_q5_k_xl`.
- Grading: against [answer_key.md](answer_key.md). Raw answers are in `*.md`; timing and token metadata are in `*.json`.

Legend: ✅ correct · ⚠️ minor error · ❌ wrong (not a moonwalker, or a substantive fabrication) · — missing from the list

## Per-person grading

| # | Moonwalker | 12B · no think | 12B · think | 31B · no think | 31B · think | Qwen · no think | Qwen · think |
|---|---|---|---|---|---|---|---|
| 1 | Neil Armstrong | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 2 | Buzz Aldrin | ✅ | ❌ "naval aviator" (he was USAF) | ✅ | ✅ | ⚠️ "developed the docking maneuver" (rendezvous techniques) | ⚠️ "test pilot" (fighter pilot) |
| 3 | Pete Conrad | ✅ | ✅ | ✅ | ✅ | — missing | ❌ "flew Apollo 8" (Gemini 11) |
| 4 | Alan Bean | ⚠️ "lunar module commander" (LMP) | ❌ "geologist by training" | ✅ | ✅ | — missing | ✅ |
| 5 | Alan Shepard | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 6 | Edgar Mitchell | ✅ | ❌ "psychologist, PhD in psychology" (MIT ScD, aeronautics) | ⚠️ epiphany "on the surface" (it was on the flight home) | ✅ | ✅ | ✅ |
| 7 | David Scott | ✅ | ❌ "a geologist" | ✅ | ✅ | ❌ "CMP of Apollo 10" (Apollo 9), "later a U.S. Congressman" | ✅ |
| 8 | James Irwin | ✅ | ✅ | ❌ "a geologist" | ✅ | — missing | ❌ "a geologist" |
| 9 | John Young | ✅ | ✅ | ❌ "flew Friendship 7" (Glenn's capsule; Young flew Gemini 3) | ✅ | ❌ "walked on the Moon twice", STS-9 "Challenger" | ❌ "veteran of Skylab" (never flew it) |
| 10 | Charles Duke | ✅ | ❌ "a geologist" | ✅ | ❌ "professional geologist" | ❌ "11th person", "NASA Administrator 1977–81" | ❌ "geologist", "first African American on the Moon", "NASA Administrator" |
| 11 | Eugene Cernan | ✅ | ✅ | ✅ | ✅ | ❌ "LMP of Apollo 17" (commander), "director of SDI office" | ⚠️ "lunar module pilot" (commander on Apollo 17) |
| 12 | Harrison Schmitt | ❌ "lunar module sample commander", "final person to walk on the Moon" (Cernan was) | ✅ | ✅ | ✅ | — missing | ✅ |
| ✗ | **Not moonwalkers, listed anyway** | none | none | none | none | ❌ Michael Collins, ❌ Jim Lovell, ❌ Fred Haise, ❌ Al Worden | none |

## Summary

| Run | Real moonwalkers listed | Non-moonwalkers listed | ✅ | ⚠️ | ❌ | Entries wrong | Background depth | Time | Tokens out |
|---|---|---|---|---|---|---|---|---|---|
| Gemma 4 12B · no think | 12/12 | 0 | 10 | 1 | 1 | **8%** | very thin (mostly mission and role) | 15 s | 427 |
| Gemma 4 12B · think | 12/12 | 0 | 7 | 0 | 5 | **42%** | moderate | 44 s | 1,228 |
| Gemma 4 31B · no think | 12/12 | 0 | 9 | 1 | 2 | **17%** | thin–moderate | 23 s | 326 |
| **Gemma 4 31B · think** | **12/12** | **0** | **11** | **0** | **1** | **8%** | moderate | 68 s | 940 |
| Qwen3.8-27B · no think | 8/12 | 4 | 3 | 1 | 8 | **67%** | rich, but largely invented | 63 s | 800 |
| Qwen3.8-27B · think | 12/12 | 0 | 6 | 2 | 4 | **33%** | rich | 595 s | 6,887 |

"Entries wrong" is ❌ ÷ 12 listed entries. A missing moonwalker isn't counted there; the "Real moonwalkers listed" column shows them.

## Observations

- **Gemma 4 31B with thinking is the clear winner.** It got 11 of 12 right with real details: Bean's painting career, Irwin's ministry, Shepard as the first American in space, Young flying both Apollo and the Shuttle. Its only error was calling Duke a geologist.
- **The 12B with thinking off scores well only because it says almost nothing.** Most entries just restate the mission role, so there is little to get wrong. When it does add background (thinking on), 5 of 12 entries are fabricated. That matches what you saw before.
- **"X was a geologist" is the most common hallucination.** It appears 6 times across 4 runs (Bean, Scott, Irwin, Duke ×2, and Irwin again). The models blur Schmitt, the one real geologist, into his colleagues.
- **Qwen3.8-27B is the weakest on this recall task.** With thinking off it got the roster itself wrong: 4 orbit-only or Apollo 13 astronauts in, 4 real moonwalkers out. It also invented career facts, such as Duke as NASA Administrator, Scott as a Congressman, and Cernan running the SDI office. Thinking fixes the roster but still leaves 4 wrong entries, and it took about 10 minutes (6.9k tokens). The model card advertises coding and agentic strengths, not trivia recall.
- **Thinking helps the bigger models but hurt the 12B.** The 12B's reasoning added more claims than it could support.

Caveat: one sample per configuration at temperature 0.7. The rankings are indicative, not statistically settled; repeating each configuration 3–5 times would firm them up.
