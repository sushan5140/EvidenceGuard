# EvidenceGuard failure analysis — frozen RAMDocs run

This report is generated automatically from the same frozen run as the result tables.
Document-type labels are used only after inference for evaluation.

## Aggregate categories

| Category | Count | Rate |
|---|---:|---:|
| strict correct | 74 | 14.8% |
| wrong answer hits | 66 | 13.2% |
| abstentions | 189 | 37.8% |
| conflict misses | 96 | 19.2% |
| improvements over hybrid | 158 | 31.6% |
| regressions vs hybrid | 46 | 9.2% |

## Wrong-answer adoption

### Case 4

**Question:** What sport is Ryan Davis associated with?

**Gold answers:** AFL

**Wrong answers:** NBA

**Outcome:** strict_correct=False, all_gold=True, any_gold=True, wrong_hit=True, abstained=False, confidence=0.819, conflict_detected=False

### Case 8

**Question:** What sport is Justin Thomas associated with?

**Gold answers:** American football

**Wrong answers:** Golf

**Outcome:** strict_correct=False, all_gold=False, any_gold=False, wrong_hit=True, abstained=False, confidence=0.656, conflict_detected=False

### Case 14

**Question:** Who is the artist of the album "A Kind of Hush"?

**Gold answers:** Carpenters

**Wrong answers:** The Beatles

**Outcome:** strict_correct=False, all_gold=True, any_gold=True, wrong_hit=True, abstained=False, confidence=0.651, conflict_detected=True

### Case 19

**Question:** What sport is Ronald Powell associated with?

**Gold answers:** Football

**Wrong answers:** Chess

**Outcome:** strict_correct=False, all_gold=False, any_gold=False, wrong_hit=True, abstained=False, confidence=0.795, conflict_detected=False

### Case 21

**Question:** What is the type of institution Fontbonne is?

**Gold answers:** academy

**Wrong answers:** university, military base

**Outcome:** strict_correct=False, all_gold=False, any_gold=False, wrong_hit=True, abstained=False, confidence=0.769, conflict_detected=False


## Conflict misses

### Case 13

**Question:** Who is the artist of the album "VII"?

**Gold answers:** Blitzen Trapper

**Wrong answers:** The Beatles, The Beatles

**Outcome:** strict_correct=True, all_gold=True, any_gold=True, wrong_hit=False, abstained=False, confidence=0.745, conflict_detected=False

### Case 15

**Question:** Who is the artist of the album "The Heat"?

**Gold answers:** Needtobreathe

**Wrong answers:** Coldplay

**Outcome:** strict_correct=True, all_gold=True, any_gold=True, wrong_hit=False, abstained=False, confidence=0.785, conflict_detected=False

### Case 16

**Question:** What sport is Doak associated with?

**Gold answers:** Football

**Wrong answers:** Soccer, Chess

**Outcome:** strict_correct=False, all_gold=False, any_gold=False, wrong_hit=False, abstained=False, confidence=0.801, conflict_detected=False

### Case 21

**Question:** What is the type of institution Fontbonne is?

**Gold answers:** academy

**Wrong answers:** university, military base

**Outcome:** strict_correct=False, all_gold=False, any_gold=False, wrong_hit=True, abstained=False, confidence=0.769, conflict_detected=False

### Case 23

**Question:** Where is the Kate Chopin House located?

**Gold answers:** Cloutierville, Louisiana

**Wrong answers:** New Orleans, Louisiana

**Outcome:** strict_correct=True, all_gold=True, any_gold=True, wrong_hit=False, abstained=False, confidence=0.669, conflict_detected=False


## Cases improved relative to Hybrid RAG

### Case 3

**Question:** Who are the directors of the film "Lahu Ke Do Rang"?

**Gold answers:** Mahesh Bhatt

**Wrong answers:** Raj Kapoor, Raj Kapoor

**Outcome:** strict_correct=False, all_gold=False, any_gold=False, wrong_hit=False, abstained=True, confidence=0.313, conflict_detected=True

### Case 9

**Question:** When was General Bryan born?

**Gold answers:** February 8, 1900

**Wrong answers:** July 15, 1905

**Outcome:** strict_correct=False, all_gold=False, any_gold=False, wrong_hit=False, abstained=True, confidence=0.190, conflict_detected=True

### Case 22

**Question:** When did Louis Alexandre die?

**Gold answers:** 1 December 1737

**Wrong answers:** 15 June 1745, 1 January 1738

**Outcome:** strict_correct=False, all_gold=False, any_gold=False, wrong_hit=False, abstained=True, confidence=0.292, conflict_detected=True

### Case 33

**Question:** What is the population of Madi Municipality?

**Gold answers:** About 50,000

**Wrong answers:** About 10,000

**Outcome:** strict_correct=True, all_gold=True, any_gold=True, wrong_hit=False, abstained=False, confidence=0.672, conflict_detected=True

### Case 38

**Question:** Where was Airways International based?

**Gold answers:** Miami, Florida

**Wrong answers:** Anchorage, Alaska

**Outcome:** strict_correct=True, all_gold=True, any_gold=True, wrong_hit=False, abstained=False, confidence=0.825, conflict_detected=False


## Regressions relative to Hybrid RAG

### Case 5

**Question:** When was the Cathedral of Saint Augustine established?

**Gold answers:** 1856

**Wrong answers:** 1900

**Outcome:** strict_correct=False, all_gold=False, any_gold=False, wrong_hit=False, abstained=True, confidence=0.455, conflict_detected=True

### Case 7

**Question:** When was the Dallas County Courthouse built?

**Gold answers:** 1902

**Wrong answers:** 1885, 1850

**Outcome:** strict_correct=False, all_gold=False, any_gold=False, wrong_hit=False, abstained=True, confidence=0.336, conflict_detected=True

### Case 10

**Question:** What is the population of Pilzno?

**Gold answers:** 4,411

**Wrong answers:** 10,000

**Outcome:** strict_correct=False, all_gold=False, any_gold=False, wrong_hit=False, abstained=True, confidence=0.383, conflict_detected=True

### Case 12

**Question:** What is the length of the Moravica river?

**Gold answers:** 98 km

**Wrong answers:** n/a

**Outcome:** strict_correct=False, all_gold=False, any_gold=False, wrong_hit=False, abstained=True, confidence=0.213, conflict_detected=True

### Case 27

**Question:** What is the language of the television series "Ardhangini"?

**Gold answers:** Assamese-language

**Wrong answers:** Finnish-language

**Outcome:** strict_correct=False, all_gold=False, any_gold=False, wrong_hit=False, abstained=True, confidence=0.316, conflict_detected=True

