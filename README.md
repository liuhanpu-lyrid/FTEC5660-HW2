# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:
> to students: this is your report, see the homework description.

### Task 1
> one paragraph describing your solution, and your results on `public_test`.
I implemented a two-stage CV verification system using LangChain and the DeepSeek V4 Flash model. In the first stage, the PDF content is extracted using MarkItDown and processed by an LLM-based parser, which converts the candidate's personal information, employment history, education, and skills into structured JSON. In the second stage, a verification agent uses the SocialGraph MCP tools to identify the correct candidate and compare the extracted claims against their LinkedIn and Facebook profiles. LinkedIn serves as the primary source of evidence, while Facebook provides additional information for identity verification. The agent checks factual discrepancies in job titles, seniority, employment dates, education, location, and skills, while allowing equivalent wording and omitted information. A clear discrepancy receives a reliability score of 0.05, a fully consistent CV receives 0.95, and uncertain cases receive 0.40. To improve robustness, I implemented JSON parsing with error handling, independent exception handling for each CV, and asynchronous processing limited to three concurrent CVs. On the public test set, the agent correctly classified 7/7 CVs, achieving an accuracy of 100%.

### Task 2
> the attack techniques you used and why you expect them to work.
For Task 2, I designed an adversarial CV for Kelly Tsang using a PDF visual-text mismatch attack. According to her LinkedIn profile, her actual position at AIA is Manager (2022–Present), with mid-level seniority. In the adversarial PDF, I changed her visibly displayed position to Director, introducing a deliberate false claim while keeping her other employment records, education, and skills unchanged.
The attack exploits the difference between the visual representation of a PDF and the text extracted by a PDF parser. Specifically, the PDF contains an underlying text layer with the correct title, "Manager", while an overlaid visual layer displays the false title, "Director". Since the verification agents process text extracted by MarkItDown rather than directly interpreting the rendered PDF, they may receive the correct title instead of the false claim visible to human readers. Consequently, the agents can verify the extracted information against SocialGraph and incorrectly classify the visually falsified CV as trustworthy.
During local testing, a conventional CV containing an inflated job title was correctly identified as having a discrepancy, receiving a score of 0.05. An earlier version of the visual-text mismatch attack received a reliability score of 0.95 from my Task 1 verifier, demonstrating the effectiveness of this approach against my implementation. This attack highlights a potential vulnerability in AI document verification systems: even when the reasoning and verification processes are accurate, discrepancies between rendered document content and extracted text can undermine the reliability of the final judgment.


## Task 3: Reflection on AI Paradoxes

My reflection is available in reflection.md.