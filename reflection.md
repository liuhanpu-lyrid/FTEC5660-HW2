# Task 3: Reflection on AI Paradoxes

Working on Homework 2 changed how I think about the reliability of AI systems. In Task 1, I built an agent to verify claims in CVs against mock LinkedIn and Facebook profiles. In Task 2, I tried to make a false CV pass a similar verification process. These two tasks revealed several tensions that I see as important AI paradoxes: automation can increase efficiency while creating new dependence on human judgment; a system designed to establish trust can produce misleading assurances; and better AI reasoning does not necessarily mean better system security.

## 1. Automation versus Human Oversight

AI can make repetitive verification tasks much faster. My agent extracts information from PDFs, identifies a candidate using SocialGraph tools, compares employment and education records, and returns a reliability score. It correctly classified all seven CVs in the public test set. That result showed me the practical value of automation, but it also raised a concern: a high test accuracy can make users more willing to accept the system's output without questioning how it was produced.

The paradox is that the more convenient an automated decision becomes, the easier it is to overlook the need to supervise it. An agent might select the wrong person with a similar name, encounter incomplete profile data, or miss a claim during extraction. A numerical score cannot communicate all of those limitations by itself. For high-stakes uses such as recruitment or compliance, AI should support human judgment rather than quietly replace it. Uncertain cases need to be flagged, and users should be able to inspect the evidence behind a decision.

## 2. Verification versus Trust

A second paradox appeared in Task 2. A CV verifier exists to detect false claims, yet it can be fooled without changing its underlying factual reasoning. My adversarial PDF visibly presented Kelly Tsang's AIA position as **Director**, while the underlying extractable text retained her actual title, **Manager**. The verifier used text extracted from the document and compared that text with the mock SocialGraph profile. In a local test of an earlier version of this technique, my Task 1 verifier assigned a reliability score of **0.95**, even though the rendered CV contained an inflated job title.

This result does not necessarily mean the language model reasoned incorrectly. It may have compared the evidence accurately *for the text it received*. The failure arose earlier, when the document's visible meaning and its extracted representation diverged. This distinction changed my understanding of trustworthy AI: correct reasoning over an inaccurate representation can still produce a misleading conclusion. A trustworthy document-verification system therefore needs to verify not just claims against external records, but also that the claims it extracts match what a person actually sees.

## 3. Intelligence versus Security

It is tempting to assume that a more capable AI model will also be harder to attack. Task 2 suggests otherwise. The attack targeted the document-processing pipeline rather than trying to persuade the model with a direct instruction. Strong prompts telling an agent to ignore malicious commands cannot fully address a mismatch between a PDF's visual and text layers. Security depends on the entire workflow: file parsing, data extraction, identity matching, tool use, and final scoring.

A better design could compare rendered pages with extracted text, detect overlapping or hidden content, and send suspicious documents for additional review. Testing should also include adversarial files, not only normal examples. Importantly, one successful local test is not proof that an attack will work against every verifier; similarly, perfect accuracy on a small public dataset does not guarantee robustness on unseen inputs.

## 4. Verification Coverage versus Fairness

There is also a tension between checking more information and making fair decisions. Social profiles may be incomplete, outdated, or presented differently from a CV. For example, a skill not listed on LinkedIn is not automatically proof that a person lacks that skill in real life. This homework defines specific comparison rules for its mock dataset, but a real recruitment system must distinguish **a confirmed contradiction** from **missing evidence**. Otherwise, broad automated checking could make decisions appear objective while unfairly penalizing candidates whose public profiles are less complete.

## Conclusion

The main lesson I take from this assignment is that AI reliability is a system-level property, not just a model-level property. My Task 1 agent demonstrated that AI can efficiently verify structured claims. Task 2 showed that the same system can give a confident answer when its input representation fails to capture the document's visible content. I would therefore design future AI agents with explicit uncertainty, traceable evidence, checks between visual and extracted content, and human review for suspicious or consequential cases. The goal should not be to eliminate human judgment, but to make that judgment better informed.