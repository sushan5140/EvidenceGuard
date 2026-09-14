# EvidenceGuard failure analysis — frozen RAMDocs run

This report is generated automatically from the same frozen run as the result tables.
Document-type labels are used only after inference for evaluation.

## Aggregate categories

| Category | Count | Rate |
|---|---:|---:|
| gold hits | 234 | 46.8% |
| wrong answer hits | 65 | 13.0% |
| abstentions | 189 | 37.8% |
| conflict misses | 96 | 19.2% |
| improvements over hybrid | 98 | 19.6% |
| regressions vs hybrid | 181 | 36.2% |

## Wrong-answer adoption

### Case 4

**Question:** What sport is Ryan Davis associated with?

**Gold answers:** AFL

**Wrong answers:** NBA

**Outcome:** gold_hit=True, wrong_hit=True, abstained=False, confidence=0.819, conflict_detected=False

### Case 8

**Question:** What sport is Justin Thomas associated with?

**Gold answers:** American football

**Wrong answers:** Golf

**Outcome:** gold_hit=False, wrong_hit=True, abstained=False, confidence=0.656, conflict_detected=False

### Case 14

**Question:** Who is the artist of the album "A Kind of Hush"?

**Gold answers:** Carpenters

**Wrong answers:** The Beatles

**Outcome:** gold_hit=True, wrong_hit=True, abstained=False, confidence=0.651, conflict_detected=True

### Case 19

**Question:** What sport is Ronald Powell associated with?

**Gold answers:** Football

**Wrong answers:** Chess

**Outcome:** gold_hit=False, wrong_hit=True, abstained=False, confidence=0.795, conflict_detected=False

### Case 21

**Question:** What is the type of institution Fontbonne is?

**Gold answers:** academy

**Wrong answers:** university, military base

**Outcome:** gold_hit=False, wrong_hit=True, abstained=False, confidence=0.769, conflict_detected=False


## Conflict misses

### Case 13

**Question:** Who is the artist of the album "VII"?

**Gold answers:** Blitzen Trapper

**Wrong answers:** The Beatles, The Beatles

**Outcome:** gold_hit=True, wrong_hit=False, abstained=False, confidence=0.745, conflict_detected=False

### Case 15

**Question:** Who is the artist of the album "The Heat"?

**Gold answers:** Needtobreathe

**Wrong answers:** Coldplay

**Outcome:** gold_hit=True, wrong_hit=False, abstained=False, confidence=0.785, conflict_detected=False

### Case 16

**Question:** What sport is Doak associated with?

**Gold answers:** Football

**Wrong answers:** Soccer, Chess

**Outcome:** gold_hit=False, wrong_hit=False, abstained=False, confidence=0.801, conflict_detected=False

### Case 21

**Question:** What is the type of institution Fontbonne is?

**Gold answers:** academy

**Wrong answers:** university, military base

**Outcome:** gold_hit=False, wrong_hit=True, abstained=False, confidence=0.769, conflict_detected=False

### Case 23

**Question:** Where is the Kate Chopin House located?

**Gold answers:** Cloutierville, Louisiana

**Wrong answers:** New Orleans, Louisiana

**Outcome:** gold_hit=True, wrong_hit=False, abstained=False, confidence=0.669, conflict_detected=False


## Cases improved relative to Hybrid RAG

### Case 3

**Question:** Who are the directors of the film "Lahu Ke Do Rang"?

**Gold answers:** Mahesh Bhatt

**Wrong answers:** Raj Kapoor, Raj Kapoor

**Outcome:** gold_hit=False, wrong_hit=False, abstained=True, confidence=0.313, conflict_detected=True

### Case 4

**Question:** What sport is Ryan Davis associated with?

**Gold answers:** AFL

**Wrong answers:** NBA

**Outcome:** gold_hit=True, wrong_hit=True, abstained=False, confidence=0.819, conflict_detected=False

### Case 9

**Question:** When was General Bryan born?

**Gold answers:** February 8, 1900

**Wrong answers:** July 15, 1905

**Outcome:** gold_hit=False, wrong_hit=False, abstained=True, confidence=0.190, conflict_detected=True

### Case 14

**Question:** Who is the artist of the album "A Kind of Hush"?

**Gold answers:** Carpenters

**Wrong answers:** The Beatles

**Outcome:** gold_hit=True, wrong_hit=True, abstained=False, confidence=0.651, conflict_detected=True

### Case 25

**Question:** What is the medium of "Loitering with Intent"?

**Gold answers:** novel

**Wrong answers:** film, film

**Outcome:** gold_hit=True, wrong_hit=True, abstained=False, confidence=0.750, conflict_detected=False


## Regressions relative to Hybrid RAG

### Case 3

**Question:** Who are the directors of the film "Lahu Ke Do Rang"?

**Gold answers:** Mahesh Bhatt

**Wrong answers:** Raj Kapoor, Raj Kapoor

**Outcome:** gold_hit=False, wrong_hit=False, abstained=True, confidence=0.313, conflict_detected=True

### Case 5

**Question:** When was the Cathedral of Saint Augustine established?

**Gold answers:** 1856

**Wrong answers:** 1900

**Outcome:** gold_hit=False, wrong_hit=False, abstained=True, confidence=0.455, conflict_detected=True

### Case 7

**Question:** When was the Dallas County Courthouse built?

**Gold answers:** 1902

**Wrong answers:** 1885, 1850

**Outcome:** gold_hit=False, wrong_hit=False, abstained=True, confidence=0.336, conflict_detected=True

### Case 9

**Question:** When was General Bryan born?

**Gold answers:** February 8, 1900

**Wrong answers:** July 15, 1905

**Outcome:** gold_hit=False, wrong_hit=False, abstained=True, confidence=0.190, conflict_detected=True

### Case 10

**Question:** What is the population of Pilzno?

**Gold answers:** 4,411

**Wrong answers:** 10,000

**Outcome:** gold_hit=False, wrong_hit=False, abstained=True, confidence=0.383, conflict_detected=True

