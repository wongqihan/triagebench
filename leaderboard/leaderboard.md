# TriageBench leaderboard

*Snapshot 2026-06-15. Cell = **TriageGap** in points (gap in ER-referral rate across the swapped attribute). **Lower = more consistent.** **Bold** = significant at p<0.05 (two-proportion z-test). Each cell shows gap (p-value).*

| Probe \ Model | claude-haiku-4-5 | claude-opus-4-8 | deepseek-v4-flash | deepseek-v4-pro | gemini-3.1-flash-lite | gemini-3.1-pro-preview | gpt-5.4-nano | gpt-5.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Language (English vs Japanese) | 0.0 (p=1.000) | **50.0 (p=0.000)** | 3.3 (p=0.313) | **26.7 (p=0.028)** | 3.3 (p=0.313) | 0.0 (p=1.000) | **46.7 (p=0.000)** | 0.0 (p=1.000) |
| Gender (man vs woman) | 0.0 (p=1.000) | **30.0 (p=0.005)** | 6.7 (p=0.584) | **23.3 (p=0.020)** | 0.0 (p=1.000) | 0.0 (p=1.000) | **33.3 (p=0.006)** | 0.0 (p=1.000) |
| SES (rich vs poor ZIP) | 0.0 (p=1.000) | 0.0 (p=1.000) | 6.9 (p=0.143) | 6.7 (p=0.573) | 3.3 (p=0.554) | 0.0 (p=1.000) | 13.3 (p=0.222) | 0.0 (p=1.000) |

*n=30 per condition. TriageGap = max-min ER-rate spread across the attribute's levels.*