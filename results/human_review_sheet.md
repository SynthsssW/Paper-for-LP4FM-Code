# Human review sheet — corpus paraphrase check (100/100 problems)

All 100 problems passed mechanical verification (see `results/corpus_verification.md`) — nothing was dropped. This sheet is for the one judgment call the verifier does not make: **does `para` still mean the same thing as `base`?** Read `base` then `para` directly beneath it. `formal` is the notation-change condition (no derivation, not executable). `break` is the answer-changing control — its changed answer is shown so you can confirm at a glance that it really did flip.

---

### 1. `gsms_0000` — answer: **2**

**base**  Amara saw a 200-foot dolphin with 8 6-inch remoras attached to it. What percentage of the dolphin's body length is the combined length of the remoras?
**para**  Attached to the 200-foot dolphin that Amara saw were 8 remoras, each of them 6 inches long. The combined length of those remoras is what percentage of the dolphin's body length?  <sub>(passive_voice)</sub>
**formal**  GIVEN x=200, k=8, y=6 — WHERE: Amara saw a x-foot dolphin with k y-inch remoras attached to it. — FIND: What percentage of the dolphin's body length is the combined length of the remoras? → `answer = ?`
**break**  (x: 200→100) Amara saw a 100-foot dolphin with 8 6-inch remoras attached to it. What percentage of the dolphin's body length is the combined length of the remoras?  → answer: **4** (was 2)

---

### 2. `gsms_0001` — answer: **296**

**base**  A fog bank rolls in from the ocean to cover a city. It takes 148 minutes to cover every 16 miles of the city. If the city is 32 miles across from the oceanfront to the opposite inland edge, how many minutes will it take for the fog bank to cover the whole city?
**para**  A city measures 32 miles from its oceanfront to its opposite inland edge, and a fog bank is rolling in off the ocean to cover it. Every 16 miles of the city take the fog 148 minutes to cover. How long, in minutes, before the whole city is covered?  <sub>(clause_reorder)</sub>
**formal**  GIVEN t=148, d=16, y=32 — WHERE: A fog bank rolls in from the ocean to cover a city. It takes t minutes to cover every d miles of the city. — FIND: If the city is y miles across from the oceanfront to the opposite inland edge, how many minutes will it take for the fog bank to cover the whole city? → `answer = ?`
**break**  (t: 148→147) A fog bank rolls in from the ocean to cover a city. It takes 147 minutes to cover every 16 miles of the city. If the city is 32 miles across from the oceanfront to the opposite inland edge, how many minutes will it take for the fog bank to cover the whole city?  → answer: **294** (was 296)

---

### 3. `gsms_0002` — answer: **25**

**base**  Luis is rolling a twelve-sided die. How much more likely is it (expressed as a percentage) that he rolls a number greater than 6 than that he rolls two odd numbers in a row?
**para**  Luis rolls a twelve-sided die. Expressed as a percentage, by how much does the chance of his rolling a number greater than 6 exceed the chance of his rolling two odd numbers in a row?  <sub>(nominalization)</sub>
**formal**  GIVEN sides=12, target=6 — WHERE: Luis is rolling a sides die. — FIND: How much more likely is it (expressed as a percentage) that he rolls a number greater than target than that he rolls two odd numbers in a row? → `answer = ?`
**break**  (target: 6→3) Luis is rolling a twelve-sided die. How much more likely is it (expressed as a percentage) that he rolls a number greater than 3 than that he rolls two odd numbers in a row?  → answer: **50** (was 25)

---

### 4. `gsms_0003` — answer: **35**

**base**  A group of 178 students has various hobbies. 32 like to hike, 37 like to play tennis, and the rest like to either paint or dance. How many like to paint if the number that like to dance is twice the number that prefer playing tennis?
**para**  In a group of 178 students there are various hobbies: 32 of them like to hike, 37 like to play tennis, and everyone else likes either to paint or to dance. If the number who like to dance is twice the number who prefer playing tennis, how many like to paint?  <sub>(fronted_conditional)</sub>
**formal**  GIVEN n=178, n_1=32, n_2=37 — WHERE: A group of n students has various hobbies. n_1 like to hike, n_2 like to play tennis, and the rest like to either paint or dance. — FIND: How many like to paint if the number that like to dance is twice the number that prefer playing tennis? → `answer = ?`
**break**  (n: 178→177) A group of 177 students has various hobbies. 32 like to hike, 37 like to play tennis, and the rest like to either paint or dance. How many like to paint if the number that like to dance is twice the number that prefer playing tennis?  → answer: **34** (was 35)

---

### 5. `gsms_0004` — answer: **765**

**base**  Marcus is popping popcorn for a snack. As the pot of kernels heats up, the kernels start popping faster. 60 pop in the first 12 seconds of cooking, then 2 times that amount in the next 12 seconds. The kernels increase to 6 times the initial popping rate in the next 12 seconds, but in the final 12 seconds, the popping slows down to half the rate as the past 12 seconds. After Marcus takes the pot off the heat, a quarter of the number of kernels that popped in the final 12 seconds of cooking also pop from the residual heat. How many pieces of popcorn does Marcus have to eat?
**para**  Marcus is making popcorn for a snack, and the kernels pop faster as the pot holding them heats up. In the first 12 seconds of cooking 60 kernels pop, and in the next 12 seconds 2 times that amount pops. Over the following 12 seconds the rate climbs to 6 times the initial popping rate, but across the last 12 seconds it falls to half the rate of the preceding 12 seconds. Once Marcus lifts the pot off the heat, residual heat pops a further quarter of the number that popped during those final 12 seconds. How many pieces of popcorn does Marcus have to eat?  <sub>(clause_reorder)</sub>
**formal**  GIVEN n=60, s=12, k_2=2, k_3=6 — WHERE: Marcus is popping popcorn for a snack. As the pot of kernels heats up, the kernels start popping faster. n pop in the first s seconds of cooking, then k_2 times that amount in the next s seconds. The kernels increase to k_3 times the initial popping rate in the next s seconds, but in the final s seconds, the popping slows down to half the rate as the past s seconds. After Marcus takes the pot off the heat, a quarter of the number of kernels that popped in the final s seconds of cooking also pop from the residual heat. — FIND: How many pieces of popcorn does Marcus have to eat? → `answer = ?`
**break**  (n: 60→40) Marcus is popping popcorn for a snack. As the pot of kernels heats up, the kernels start popping faster. 40 pop in the first 12 seconds of cooking, then 2 times that amount in the next 12 seconds. The kernels increase to 6 times the initial popping rate in the next 12 seconds, but in the final 12 seconds, the popping slows down to half the rate as the past 12 seconds. After Marcus takes the pot off the heat, a quarter of the number of kernels that popped in the final 12 seconds of cooking also pop from the residual heat. How many pieces of popcorn does Marcus have to eat?  → answer: **510** (was 765)

---

### 6. `gsms_0005` — answer: **148**

**base**  Elise makes tea using teaspoons of sugar and cups of water in the ratio of 148:75. If she used a total of 223 teaspoons of sugar and cups of water, calculate the number of teaspoonfuls of sugar she used.
**para**  Elise's recipe for tea calls for teaspoons of sugar and cups of water in the ratio 148:75. She used 223 teaspoons of sugar and cups of water in total. How many teaspoonfuls of sugar did she use?  <sub>(imperative_to_wh)</sub>
**formal**  GIVEN m=148, n=75, x=223 — WHERE: Elise makes tea using teaspoons of sugar and cups of water in the ratio of m:n. — FIND: If she used a total of x teaspoons of sugar and cups of water, calculate the number of teaspoonfuls of sugar she used. → `answer = ?`
**break**  (m: 148→147) Elise makes tea using teaspoons of sugar and cups of water in the ratio of 147:75. If she used a total of 223 teaspoons of sugar and cups of water, calculate the number of teaspoonfuls of sugar she used.  → answer: **147** (was 148)

---

### 7. `gsms_0006` — answer: **1300**

**base**  Robert has 25 square yards of peach field. There are 13 peachs per square yard. Robert can harvest his peachs every 3 months. How many peachs can Robert harvest within a year?
**para**  Robert owns a peach field of 25 square yards, each square yard carrying 13 peachs, and he is able to harvest his peachs once every 3 months. Within a year, how many peachs can Robert harvest?  <sub>(sentence_join)</sub>
**formal**  GIVEN x=25, k=13, n=3 — WHERE: Robert has x square yards of peach field. There are k peachs per square yard. Robert can harvest his peachs every n months. — FIND: How many peachs can Robert harvest within a year? → `answer = ?`
**break**  (x: 25→20) Robert has 20 square yards of peach field. There are 13 peachs per square yard. Robert can harvest his peachs every 3 months. How many peachs can Robert harvest within a year?  → answer: **1040** (was 1300)

---

### 8. `gsms_0007` — answer: **18**

**base**  A farmer is buying feed for his sheep. He buys a variety of hay, oats, carrots and apples. Since apples are a rare treat, he only buys four 5-kg boxes of them for the whole stable. He only wants enough carrots to feed the sheep while the vegetables are fresh, so he buys twelve 9-kg bags. Hay are the main diet of his sheep, so he buys thirty four 38-kg packs. Oats are a staple to supplement the hay, so he buys twenty 19-kg sacks. If his farm truck can carry 100 kgs at a time, how many trips does the farmer need to transport all the feed?
**para**  A farmer buying feed for his sheep picks up a variety of hay, oats, carrots and apples. Because hay are what his sheep mostly eat, he takes thirty four 38-kg packs of it, and to supplement that he takes twenty 19-kg sacks of oats. He wants only as much carrots as the sheep can get through while the vegetables are fresh, so he takes twelve 9-kg bags, and since apples are a rare treat, only four 5-kg boxes for the whole stable. His farm truck carries 100 kgs per trip. How many trips does the farmer need to transport all the feed?  <sub>(enumeration_reorder)</sub>
**formal**  GIVEN n4=4, w4=5, n3=12, w3=9, n1=34, w1=38, n2=20, w2=19, total=100 — WHERE: A farmer is buying feed for his sheep. He buys a variety of hay, oats, carrots and apples. Since apples are a rare treat, he only buys n4 w4-kg boxes of them for the whole stable. He only wants enough carrots to feed the sheep while the vegetables are fresh, so he buys n3 w3-kg bags. Hay are the main diet of his sheep, so he buys n1 w1-kg packs. Oats are a staple to supplement the hay, so he buys n2 w2-kg sacks. — FIND: If his farm truck can carry total kgs at a time, how many trips does the farmer need to transport all the feed? → `answer = ?`
**break**  (total: 100→120) A farmer is buying feed for his sheep. He buys a variety of hay, oats, carrots and apples. Since apples are a rare treat, he only buys four 5-kg boxes of them for the whole stable. He only wants enough carrots to feed the sheep while the vegetables are fresh, so he buys twelve 9-kg bags. Hay are the main diet of his sheep, so he buys thirty four 38-kg packs. Oats are a staple to supplement the hay, so he buys twenty 19-kg sacks. If his farm truck can carry 120 kgs at a time, how many trips does the farmer need to transport all the feed?  → answer: **15** (was 18)

---

### 9. `gsms_0008` — answer: **84**

**base**  A juggler can juggle 840 balls. A fifth of the balls are tennis balls, and half of the tennis balls are orange. How many orange tennis balls are there?
**para**  840 balls can be juggled by a juggler. A fifth of them are tennis balls, and of those tennis balls, half are orange. How many orange tennis balls are there?  <sub>(passive_voice)</sub>
**formal**  GIVEN n=840, frac_1=Fraction(1, 5), frac_2=Fraction(1, 2) — WHERE: A juggler can juggle n balls. frac_1 of the balls are tennis balls, and frac_2 of the tennis balls are orange. — FIND: How many orange tennis balls are there? → `answer = ?`
**break**  (n: 840→820) A juggler can juggle 820 balls. A fifth of the balls are tennis balls, and half of the tennis balls are orange. How many orange tennis balls are there?  → answer: **82** (was 84)

---

### 10. `gsms_0009` — answer: **95**

**base**  When Mei watches her nephew, she gets out a variety of toys for him. The bag of building blocks has 72 blocks in it. The bin of stuffed animals has 40 stuffed animals inside. The tower of stacking rings has 32 multicolored rings on it. Mei recently bought a tube of bouncy balls, bringing her total number of toys for her nephew up to 239. How many bouncy balls came in the tube?
**para**  Mei gets out a variety of toys whenever she watches her nephew: a bag holding 72 building blocks, a bin holding 40 stuffed animals, and a tower holding 32 multicolored rings. A tube of bouncy balls that Mei bought recently brought her total number of toys for her nephew up to 239. How many bouncy balls were in the tube?  <sub>(sentence_join)</sub>
**formal**  GIVEN x=72, y=40, z=32, total=239 — WHERE: When Mei watches her nephew, she gets out a variety of toys for him. The bag of building blocks has x blocks in it. The bin of stuffed animals has y stuffed animals inside. The tower of stacking rings has z multicolored rings on it. Mei recently bought a tube of bouncy balls, bringing her total number of toys for her nephew up to total. — FIND: How many bouncy balls came in the tube? → `answer = ?`
**break**  (x: 72→71) When Mei watches her nephew, she gets out a variety of toys for him. The bag of building blocks has 71 blocks in it. The bin of stuffed animals has 40 stuffed animals inside. The tower of stacking rings has 32 multicolored rings on it. Mei recently bought a tube of bouncy balls, bringing her total number of toys for her nephew up to 239. How many bouncy balls came in the tube?  → answer: **96** (was 95)

---

### 11. `gsms_0010` — answer: **19**

**base**  To make a call from a hotel room phone, you must pay $0.5 for each minute of your call. After 10 minutes, that price drops to $0.25 per minute. How much would a 66-minute call cost?
**para**  Calls from a hotel room phone cost $0.5 for each minute, but once a call has run 10 minutes the rate drops to $0.25 per minute. What would a 66-minute call cost?  <sub>(topicalization)</sub>
**formal**  GIVEN frac1=Fraction(1, 2), t=10, frac2=Fraction(1, 4), total=66 — WHERE: To make a call from a hotel room phone, you must pay $frac1 for each minute of your call. After t minutes, that price drops to $frac2 per minute. — FIND: How much would a total-minute call cost? → `answer = ?`
**break**  (t: 10→14) To make a call from a hotel room phone, you must pay $0.5 for each minute of your call. After 14 minutes, that price drops to $0.25 per minute. How much would a 66-minute call cost?  → answer: **20** (was 19)

---

### 12. `gsms_0011` — answer: **72**

**base**  Ali wants to make different sized ice cubes with 344 ounces of water. He can make giant cubes that use 13 ounces per cube, medium cubes that use 11 ounces, and small cubes that use 1/2 of ounce. If he makes 9 giant cubes, 24 small cubes, and 13 medium cubes, how many ounces of water does he have left?
**para**  Ali has 344 ounces of water and wants to make ice cubes in different sizes. A giant cube uses 13 ounces, a medium cube uses 11 ounces, and a small cube uses 1/2 of an ounce. He makes 9 giant cubes, 13 medium cubes and 24 small cubes. How many ounces of water does he have left?  <sub>(sentence_split)</sub>
**formal**  GIVEN n=344, w1=13, w2=11, w3=Fraction(1, 2), n1=9, n3=24, n2=13 — WHERE: Ali wants to make different sized ice cubes with n ounces of water. He can make giant cubes that use w1 ounces per cube, medium cubes that use w2 ounces, and small cubes that use w3 of ounce. — FIND: If he makes n1 giant cubes, n3 small cubes, and n2 medium cubes, how many ounces of water does he have left? → `answer = ?`
**break**  (n: 344→342) Ali wants to make different sized ice cubes with 342 ounces of water. He can make giant cubes that use 13 ounces per cube, medium cubes that use 11 ounces, and small cubes that use 1/2 of ounce. If he makes 9 giant cubes, 24 small cubes, and 13 medium cubes, how many ounces of water does he have left?  → answer: **70** (was 72)

---

### 13. `gsms_0012` — answer: **51**

**base**  Michael places olives on some tables. Each table can hold 23 olives. If he has 304 olives and 11 tables, how many olives won't he be able to place on the table?
**para**  Michael has 304 olives to be placed on 11 tables, and each table holds 23 olives. How many olives will be left unplaced?  <sub>(passive_voice)</sub>
**formal**  GIVEN x=23, n=304, k=11 — WHERE: Michael places olives on some tables. Each table can hold x olives. — FIND: If he has n olives and k tables, how many olives won't he be able to place on the table? → `answer = ?`
**break**  (x: 23→22) Michael places olives on some tables. Each table can hold 22 olives. If he has 304 olives and 11 tables, how many olives won't he be able to place on the table?  → answer: **62** (was 51)

---

### 14. `gsms_0013` — answer: **643**

**base**  Linda is an avid gardener. Two days ago, she received 40 new potted plants from her favorite plant nursery. She already has 12 potted plants on each of the 67 window ledges of her large country home. Feeling generous, she has decided that she will give 3 potted plants from each ledge to friends and family tomorrow. How many potted plants will Linda remain with?
**para**  Two days ago, Linda, who is an avid gardener, received 40 new potted plants from her favorite plant nursery. The 67 window ledges of her large country home already hold 12 potted plants each. Tomorrow, feeling generous, she will give 3 potted plants from each ledge to friends and family. How many potted plants will Linda be left with?  <sub>(relative_clause)</sub>
**formal**  GIVEN x=40, r=12, w=67, n=3 — WHERE: Linda is an avid gardener. Two days ago, she received x new potted plants from her favorite plant nursery. She already has r potted plants on each of the w window ledges of her large country home. Feeling generous, she has decided that she will give n potted plants from each ledge to friends and family tomorrow. — FIND: How many potted plants will Linda remain with? → `answer = ?`
**break**  (x: 40→41) Linda is an avid gardener. Two days ago, she received 41 new potted plants from her favorite plant nursery. She already has 12 potted plants on each of the 67 window ledges of her large country home. Feeling generous, she has decided that she will give 3 potted plants from each ledge to friends and family tomorrow. How many potted plants will Linda remain with?  → answer: **644** (was 643)

---

### 15. `gsms_0014` — answer: **36**

**base**  Wei and Luis have 76 minutes to walk to the bank together. It takes them 13 minutes to get to the corner where the cinema is. It takes them another 27 minutes to get to the library. How much longer do they have to get to bank without being late?
**para**  Wei and Luis set out together for the bank with 76 minutes in hand. Reaching the corner where the cinema stands takes them 13 minutes, and another 27 minutes brings them to the library. How much longer do they have to get to the bank without being late?  <sub>(nominalization)</sub>
**formal**  GIVEN t=76, t1=13, t2=27 — WHERE: Wei and Luis have t minutes to walk to the bank together. It takes them t1 minutes to get to the corner where the cinema is. It takes them another t2 minutes to get to the library. — FIND: How much longer do they have to get to bank without being late? → `answer = ?`
**break**  (t: 76→77) Wei and Luis have 77 minutes to walk to the bank together. It takes them 13 minutes to get to the corner where the cinema is. It takes them another 27 minutes to get to the library. How much longer do they have to get to bank without being late?  → answer: **37** (was 36)

---

### 16. `gsms_0015` — answer: **25**

**base**  Robert's dog has 1015 puppies, 240 of which have spots. Olivia's dog has 465 puppies, 130 of which have spots. What percentage of all the puppies have spots?
**para**  Of the 1015 puppies that Robert's dog has, 240 have spots, while of the 465 that Olivia's dog has, 130 do. What percentage of all the puppies have spots?  <sub>(sentence_join)</sub>
**formal**  GIVEN n1=1015, k1=240, n2=465, k2=130 — WHERE: Robert's dog has n1 puppies, k1 of which have spots. Olivia's dog has n2 puppies, k2 of which have spots. — FIND: What percentage of all the puppies have spots? → `answer = ?`
**break**  (n1: 1015→1025) Robert's dog has 1025 puppies, 240 of which have spots. Olivia's dog has 465 puppies, 130 of which have spots. What percentage of all the puppies have spots?  → answer: **24** (was 25)

---

### 17. `gsms_0016` — answer: **1396**

**base**  Ahmed is selling cupcake and apple pie for a fundraiser. On Sunday, he sold 22 boxes of cupcake and 17 fewer boxes of apple pie, than on Monday. On Monday, he sold 14 more boxes of cupcakes than on Sunday and 26 boxes of apple pie. If the cupcake cost $12 and the apple pie cost $20, how much did Ahmed earn for two days?
**para**  Ahmed is running a fundraiser selling cupcake and apple pie. His Sunday sales came to 22 boxes of cupcake, and 17 fewer boxes of apple pie than he sold on Monday. On Monday he sold 14 more boxes of cupcakes than on Sunday, along with 26 boxes of apple pie. With cupcake priced at $12 and apple pie at $20, how much did Ahmed earn over the two days?  <sub>(clause_reorder)</sub>
**formal**  GIVEN n1=22, m1=17, n2=14, m2=26, cn=12, cm=20 — WHERE: Ahmed is selling cupcake and apple pie for a fundraiser. On Sunday, he sold n1 boxes of cupcake and m1 fewer boxes of apple pie, than on Monday. On Monday, he sold n2 more boxes of cupcakes than on Sunday and m2 boxes of apple pie. — FIND: If the cupcake cost $cn and the apple pie cost $cm, how much did Ahmed earn for two days? → `answer = ?`
**break**  (n1: 22→21) Ahmed is selling cupcake and apple pie for a fundraiser. On Sunday, he sold 21 boxes of cupcake and 17 fewer boxes of apple pie, than on Monday. On Monday, he sold 14 more boxes of cupcakes than on Sunday and 26 boxes of apple pie. If the cupcake cost $12 and the apple pie cost $20, how much did Ahmed earn for two days?  → answer: **1372** (was 1396)

---

### 18. `gsms_0017` — answer: **86**

**base**  Greta walks 23 kms every day on her favorite walking trail, which includes 10 kms of walking on the beach and 13 kms of walking on the sidewalk. On the sidewalk, Greta walks at three times the rate of speed that she does on the beach. If 60 minutes of her walk is spent on the beach, how long does it take for her to complete the entire 23-km walk, in minutes?
**para**  Every day Greta walks 23 kms along her favorite trail: 10 kms of it on the beach and 13 kms of it on the sidewalk. Her speed on the sidewalk is three times her speed on the beach. She spends 60 minutes of the walk on the beach. In minutes, how long does the whole 23-km walk take her?  <sub>(sentence_split)</sub>
**formal**  GIVEN n=23, m=10, k=13, mult=3, t=60 — WHERE: Greta walks n kms every day on her favorite walking trail, which includes m kms of walking on the beach and k kms of walking on the sidewalk. On the sidewalk, Greta walks at mult the rate of speed that she does on the beach. — FIND: If t minutes of her walk is spent on the beach, how long does it take for her to complete the entire n-km walk, in minutes? → `answer = ?`
**break**  (mult: three times→twice) Greta walks 23 kms every day on her favorite walking trail, which includes 10 kms of walking on the beach and 13 kms of walking on the sidewalk. On the sidewalk, Greta walks at twice the rate of speed that she does on the beach. If 60 minutes of her walk is spent on the beach, how long does it take for her to complete the entire 23-km walk, in minutes?  → answer: **99** (was 86)

---

### 19. `gsms_0018` — answer: **214**

**base**  Ethan went to buy some school supplies. He bought 8 rulers which cost $3.25 each, 12 notebooks which cost $14 each, and a rim of bond paper which cost $20. How much did Ethan spend on everything?
**para**  Ethan went shopping for school supplies. 8 rulers at $3.25 each, 12 notebooks at $14 each, and a rim of bond paper at $20 all went into his basket. What did Ethan spend altogether?  <sub>(topicalization)</sub>
**formal**  GIVEN n1=8, c1=3.25, n2=12, c2=14, c3=20 — WHERE: Ethan went to buy some school supplies. He bought n1 rulers which cost $c1 each, n2 notebooks which cost $c2 each, and a rim of bond paper which cost $c3. — FIND: How much did Ethan spend on everything? → `answer = ?`
**break**  (c1: 3.25→3.5) Ethan went to buy some school supplies. He bought 8 rulers which cost $3.5 each, 12 notebooks which cost $14 each, and a rim of bond paper which cost $20. How much did Ethan spend on everything?  → answer: **216** (was 214)

---

### 20. `gsms_0019` — answer: **4**

**base**  In a 20-item quiz, 40% of the questions are geometry, and the rest are equally divided as biology and chemistry questions. If Thandiwe is sure to get 25% of the geometry questions, and a sixth of the biology and chemistry questions correctly, how many points is she sure to get?
**para**  A quiz has 20 items. 40% of its questions are geometry, and the remainder are divided equally between biology and chemistry. Thandiwe is certain to answer 25% of the geometry questions correctly, together with a sixth of the biology and chemistry questions. How many points is she certain to get?  <sub>(sentence_split)</sub>
**formal**  GIVEN n=20, p1=40, r1=25, frac=Fraction(1, 6) — WHERE: In a n-item quiz, p1% of the questions are geometry, and the rest are equally divided as biology and chemistry questions. — FIND: If Thandiwe is sure to get r1% of the geometry questions, and frac of the biology and chemistry questions correctly, how many points is she sure to get? → `answer = ?`
**break**  (n: 20→10) In a 10-item quiz, 40% of the questions are geometry, and the rest are equally divided as biology and chemistry questions. If Thandiwe is sure to get 25% of the geometry questions, and a sixth of the biology and chemistry questions correctly, how many points is she sure to get?  → answer: **2** (was 4)

---

### 21. `gsms_0020` — answer: **1674**

**base**  Mateo picks 395 pineapples on Thursday. Then he picks 94 pineapples on Friday. On Saturday, he picks triple the number of pineapples he did on Thursday. How many pineapples does Mateo have?
**para**  On Thursday, 395 pineapples are picked by Mateo. On Friday he picks another 94, and on Saturday he picks triple the number he picked on Thursday. How many pineapples does Mateo have?  <sub>(passive_voice)</sub>
**formal**  GIVEN n1=395, n2=94, mult=3 — WHERE: Mateo picks n1 pineapples on Thursday. Then he picks n2 pineapples on Friday. On Saturday, he picks mult the number of pineapples he did on Thursday. — FIND: How many pineapples does Mateo have? → `answer = ?`
**break**  (n1: 395→394) Mateo picks 394 pineapples on Thursday. Then he picks 94 pineapples on Friday. On Saturday, he picks triple the number of pineapples he did on Thursday. How many pineapples does Mateo have?  → answer: **1670** (was 1674)

---

### 22. `gsms_0021` — answer: **7**

**base**  Omar watches TV after he finishes his homework every night. On Friday and Saturday, he watched a 6-hour episode of his favorite show each night. On Sunday, he watched a few episodes of a 15-minute show. On Monday, he finished homework early and watched a 2-hour episode and a 15-minute show. On Tuesday, he got to stay up late, so he watched five 3-hour episodes. If he watched 31 hours of TV in all, how many 15-minute episodes did he watch on Sunday?
**para**  Every night, once his homework is done, Omar watches TV. Each of Friday and Saturday brought a 6-hour episode of his favorite show; Sunday brought a few episodes of a 15-minute show; Monday, with homework finished early, brought a 2-hour episode plus a 15-minute show; and on Tuesday he was allowed to stay up late for five 3-hour episodes. His TV time came to 31 hours in all. How many 15-minute episodes did he watch on Sunday?  <sub>(sentence_join)</sub>
**formal**  GIVEN h1=6, m1=15, h2=2, m2=15, n2=5, h3=3, total=31 — WHERE: Omar watches TV after he finishes his homework every night. On Friday and Saturday, he watched a h1-hour episode of his favorite show each night. On Sunday, he watched a few episodes of a m1-minute show. On Monday, he finished homework early and watched a h2-hour episode and a m2-minute show. On Tuesday, he got to stay up late, so he watched n2 h3-hour episodes. — FIND: If he watched total hours of TV in all, how many m1-minute episodes did he watch on Sunday? → `answer = ?`
**break**  (h2: 2→3) Omar watches TV after he finishes his homework every night. On Friday and Saturday, he watched a 6-hour episode of his favorite show each night. On Sunday, he watched a few episodes of a 15-minute show. On Monday, he finished homework early and watched a 3-hour episode and a 15-minute show. On Tuesday, he got to stay up late, so he watched five 3-hour episodes. If he watched 31 hours of TV in all, how many 15-minute episodes did he watch on Sunday?  → answer: **3** (was 7)

---

### 23. `gsms_0022` — answer: **22**

**base**  Ines was preparing for a birthday party at her house, where she intended to serve roast chicken. She noticed that she was out of plastic spoons, so she bought a new package of spoons. Later, her husband also bought a package of 19 new spoons and gave them to Ines. While Ines was making the roast chicken, she used eight of the spoons to sample her roast chicken. Later, when she went to set the table, she had a total of 33 spoons. How many spoons were in the package that Ines bought?
**para**  Ines intended to serve roast chicken at the birthday party she was preparing at her house. Being out of plastic spoons, she bought a new package of them, and later her husband gave her another package containing 19 new spoons. While she was cooking the roast chicken, she used eight of the spoons to sample it. By the time she went to set the table she had 33 spoons altogether. How many spoons were in the package that Ines bought?  <sub>(clause_reorder)</sub>
**formal**  GIVEN n1=19, n2=8, total=33 — WHERE: Ines was preparing for a birthday party at her house, where she intended to serve roast chicken. She noticed that she was out of plastic spoons, so she bought a new package of spoons. Later, her husband also bought a package of n1 new spoons and gave them to Ines. While Ines was making the roast chicken, she used n2 of the spoons to sample her roast chicken. Later, when she went to set the table, she had a total of total spoons. — FIND: How many spoons were in the package that Ines bought? → `answer = ?`
**break**  (n1: 19→18) Ines was preparing for a birthday party at her house, where she intended to serve roast chicken. She noticed that she was out of plastic spoons, so she bought a new package of spoons. Later, her husband also bought a package of 18 new spoons and gave them to Ines. While Ines was making the roast chicken, she used eight of the spoons to sample her roast chicken. Later, when she went to set the table, she had a total of 33 spoons. How many spoons were in the package that Ines bought?  → answer: **23** (was 22)

---

### 24. `gsms_0023` — answer: **236**

**base**  Nadia is knitting winter wear for her 4 grandchildren. They're quadruplets, so they're all the same size. She wants to make a beanie, cardigan, tunic, mittens, and socks for each of them. It takes 13 skeins of wool to make a beanie, 18 for a cardigan, 12 for a tunic, 6 for a pair of mittens, and 10 for a pair of socks. How many skeins of wool will she need to buy?
**para**  Nadia's 4 grandchildren are quadruplets, so they are all the same size, and she is knitting each of them a beanie, a cardigan, a tunic, mittens, and socks for the winter. A beanie takes 13 skeins of wool, a cardigan takes 18, a tunic takes 12, a pair of mittens takes 6, and a pair of socks takes 10. How many skeins of wool will she need to buy?  <sub>(clause_reorder)</sub>
**formal**  GIVEN count=4, children=4, n1=13, n2=18, n3=12, n4=6, n5=10 — WHERE: Nadia is knitting winter wear for her count grandchildren. They're children, so they're all the same size. She wants to make a beanie, cardigan, tunic, mittens, and socks for each of them. It takes n1 skeins of wool to make a beanie, n2 for a cardigan, n3 for a tunic, n4 for a pair of mittens, and n5 for a pair of socks. — FIND: How many skeins of wool will she need to buy? → `answer = ?`
**break**  (n1: 13→12) Nadia is knitting winter wear for her 4 grandchildren. They're quadruplets, so they're all the same size. She wants to make a beanie, cardigan, tunic, mittens, and socks for each of them. It takes 12 skeins of wool to make a beanie, 18 for a cardigan, 12 for a tunic, 6 for a pair of mittens, and 10 for a pair of socks. How many skeins of wool will she need to buy?  → answer: **232** (was 236)

---

### 25. `gsms_0024` — answer: **120**

**base**  According to its nutritional info, a bag of cookies has 275 calories per serving. If a 300 ounces bag has 5 servings, how many ounces can you eat if your daily calorie target is 2100 and you have already consumed 1550 calories?
**para**  A bag of cookies contains 275 calories per serving according to its nutritional info, and one 300 ounces bag holds 5 servings. Your daily calorie target is 2100 and you have already consumed 1550 calories. How many ounces can you eat?  <sub>(sentence_split)</sub>
**formal**  GIVEN cal=275, size=300, n=5, total=2100, spent=1550 — WHERE: According to its nutritional info, a bag of cookies has cal calories per serving. — FIND: If a size ounces bag has n servings, how many ounces can you eat if your daily calorie target is total and you have already consumed spent calories? → `answer = ?`
**break**  (size: 300→350) According to its nutritional info, a bag of cookies has 275 calories per serving. If a 350 ounces bag has 5 servings, how many ounces can you eat if your daily calorie target is 2100 and you have already consumed 1550 calories?  → answer: **140** (was 120)

---

### 26. `gsms_0025` — answer: **58**

**base**  In Ms. Johnson's class of 145 students, 20% of the class are soccer players. Out of the remaining class, 25% of the students are dancers or part of debate team members. These 3 groups of students will need to leave early today to travel to an away meet. How many students are leaving early?
**para**  Ms. Johnson teaches a class of 145 students, 20% of whom are soccer players. Of the students who remain, 25% are dancers or part of debate team members. All 3 groups have to leave early today to travel to an away meet. How many students are leaving early?  <sub>(relative_clause)</sub>
**formal**  GIVEN total=145, p1=20, p2=25 — WHERE: In Ms. Johnson's class of total students, p1% of the class are soccer players. Out of the remaining class, p2% of the students are dancers or part of debate team members. These 3 groups of students will need to leave early today to travel to an away meet. — FIND: How many students are leaving early? → `answer = ?`
**break**  (total: 145→140) In Ms. Johnson's class of 140 students, 20% of the class are soccer players. Out of the remaining class, 25% of the students are dancers or part of debate team members. These 3 groups of students will need to leave early today to travel to an away meet. How many students are leaving early?  → answer: **56** (was 58)

---

### 27. `gsms_0026` — answer: **98**

**base**  An uncle eats 8 tacos per day, his partner eats 4 tacos per day, and their grandchild eats 2 tacos per day. How many tacos does this family eat in one week?
**para**  An uncle gets through 8 tacos in a day, his partner gets through 4, and their grandchild gets through 2. Over one week, how many tacos does this family eat?  <sub>(lexical_substitution)</sub>
**formal**  GIVEN n1=8, n2=4, n3=2 — WHERE: An uncle eats n1 tacos per day, his partner eats n2 tacos per day, and their grandchild eats n3 tacos per day. — FIND: How many tacos does this family eat in one week? → `answer = ?`
**break**  (n1: 8→7) An uncle eats 7 tacos per day, his partner eats 4 tacos per day, and their grandchild eats 2 tacos per day. How many tacos does this family eat in one week?  → answer: **91** (was 98)

---

### 28. `gsms_0027` — answer: **648**

**base**  Ahmed receives 144 emails a day. 25% of those emails don't require any response. Ahmed responds to the rest of them. How many emails does Ahmed respond to in a 6 day work week?
**para**  Ahmed receives 144 emails a day, of which 25% require no response at all; the rest he answers. Over a 6 day work week, how many emails does Ahmed answer?  <sub>(sentence_join)</sub>
**formal**  GIVEN n=144, p=25, d=6 — WHERE: Ahmed receives n emails a day. p% of those emails don't require any response. Ahmed responds to the rest of them. — FIND: How many emails does Ahmed respond to in a d day work week? → `answer = ?`
**break**  (n: 144→140) Ahmed receives 140 emails a day. 25% of those emails don't require any response. Ahmed responds to the rest of them. How many emails does Ahmed respond to in a 6 day work week?  → answer: **630** (was 648)

---

### 29. `gsms_0028` — answer: **1000**

**base**  A clinic has a capacity of 1600 wards with 1/5 occupied. Due to the natural disaster, 20 patients are admitted into the clinic each day. Calculate the total number of unoccupied wards in the clinic after 2 weeks.
**para**  1/5 of the 1600 wards a clinic can hold are occupied. The natural disaster is bringing 20 new patients into the clinic every day. After 2 weeks, how many wards in the clinic will be unoccupied?  <sub>(imperative_to_wh)</sub>
**formal**  GIVEN total=1600, frac=Fraction(1, 5), daily=20, period=2 — WHERE: A clinic has a capacity of total wards with frac occupied. Due to the natural disaster, daily patients are admitted into the clinic each day. — FIND: Calculate the total number of unoccupied wards in the clinic after period weeks. → `answer = ?`
**break**  (total: 1600→1500) A clinic has a capacity of 1500 wards with 1/5 occupied. Due to the natural disaster, 20 patients are admitted into the clinic each day. Calculate the total number of unoccupied wards in the clinic after 2 weeks.  → answer: **920** (was 1000)

---

### 30. `gsms_0029` — answer: **39**

**base**  James is 22 years old. In 5 years his sister Mary will be 2 times as old as James is now. How old is Mary right now?
**para**  Mary, the sister of James, will in 5 years be 2 times as old as James is now, and James is 22 years old today. How old is Mary right now?  <sub>(clause_reorder)</sub>
**formal**  GIVEN age1=22, years=5, mult=2 — WHERE: James is age1 years old. In years years his sister Mary will be mult times as old as James is now. — FIND: How old is Mary right now? → `answer = ?`
**break**  (age1: 22→21) James is 21 years old. In 5 years his sister Mary will be 2 times as old as James is now. How old is Mary right now?  → answer: **37** (was 39)

---

### 31. `gsms_0030` — answer: **36**

**base**  Jamal has ¥62.1 and wants to buy 29 nuts from a bin at the home improvement store. Each nut costs ¥0.9. How much money does Jamal have left after paying for the nuts?
**para**  Jamal wants 29 nuts from a bin at the home improvement store, where each nut costs ¥0.9. He has ¥62.1 with him. After paying for the nuts, how much money does Jamal have left?  <sub>(sentence_split)</sub>
**formal**  GIVEN initial_amount=62.1, quantity=29, unit_price=0.9 — WHERE: Jamal has ¥initial_amount and wants to buy quantity nuts from a bin at the home improvement store. Each nut costs ¥unit_price. — FIND: How much money does Jamal have left after paying for the nuts? → `answer = ?`
**break**  (quantity: 29→19) Jamal has ¥62.1 and wants to buy 19 nuts from a bin at the home improvement store. Each nut costs ¥0.9. How much money does Jamal have left after paying for the nuts?  → answer: **45** (was 36)

---

### 32. `gsms_0031` — answer: **33**

**base**  Olivia and Thandiwe are currently 22 years apart in age. If Olivia, who is younger than Thandiwe, is 22 years old, what's the average of their ages?
**para**  Olivia, who is 22 years old, is younger than Thandiwe, and the two of them are currently 22 years apart in age. What's the average of their ages?  <sub>(relative_clause)</sub>
**formal**  GIVEN age_diff=22, age1=22 — WHERE: Olivia and Thandiwe are currently age_diff years apart in age. — FIND: If Olivia, who is younger than Thandiwe, is age1 years old, what's the average of their ages? → `answer = ?`
**break**  (age_diff: 22→20) Olivia and Thandiwe are currently 20 years apart in age. If Olivia, who is younger than Thandiwe, is 22 years old, what's the average of their ages?  → answer: **32** (was 33)

---

### 33. `gsms_0032` — answer: **20400**

**base**  Ms. Johnson is looking for a house that will not go beyond her $390000 budget. She saw a property that has a selling price of $360000. On top of that, the buyer has to pay a brokerage fee which is 3% of the selling price, and also the transfer fee that is 11% of the selling price. How much more is the total price of the house than Ms. Johnson's budget?
**para**  The budget Ms. Johnson has set for a house is $390000, and she will not go beyond it. A property she saw is selling for $360000, on top of which the buyer owes a brokerage fee of 3% of the selling price and a transfer fee of 11% of the selling price. By how much does the total price of the house exceed Ms. Johnson's budget?  <sub>(nominalization)</sub>
**formal**  GIVEN budget=390000, price=360000, brokerage_fee=3, transfer_fee=11 — WHERE: Ms. Johnson is looking for a house that will not go beyond her $budget budget. She saw a property that has a selling price of $price. On top of that, the buyer has to pay a brokerage fee which is brokerage_fee% of the selling price, and also the transfer fee that is transfer_fee% of the selling price. — FIND: How much more is the total price of the house than Ms. Johnson's budget? → `answer = ?`
**break**  (budget: 390000→380000) Ms. Johnson is looking for a house that will not go beyond her $380000 budget. She saw a property that has a selling price of $360000. On top of that, the buyer has to pay a brokerage fee which is 3% of the selling price, and also the transfer fee that is 11% of the selling price. How much more is the total price of the house than Ms. Johnson's budget?  → answer: **30400** (was 20400)

---

### 34. `gsms_0033` — answer: **18**

**base**  Thandiwe and Olivia are sisters from Lakeside who love collecting signatures from politicians. During their spring break from school, the sisters spend every afternoon collecting signatures. After 6 weeks, Thandiwe and Olivia compare their autograph books, counting up the number of signatures each sister has collected. Thandiwe has 23 signatures in her book, and Olivia has 59. The sisters have 2 more weeks of spring break, and they decide they want to reach 100 signatures between them by the end of the break. How many signatures do the sisters need to collect to reach their goal?
**para**  Thandiwe and Olivia, sisters from Lakeside, love collecting signatures from politicians, and every afternoon of their spring break from school is spent collecting them. 6 weeks in, they compare autograph books: Thandiwe has 23 signatures and Olivia has 59. With 2 weeks of spring break still to go, they decide they want to reach 100 signatures between them by the end of the break. How many signatures do the sisters need to collect to reach their goal?  <sub>(sentence_join)</sub>
**formal**  GIVEN n1=6, s1=23, s2=59, n2=2, goal=100 — WHERE: Thandiwe and Olivia are sisters from Lakeside who love collecting signatures from politicians. During their spring break from school, the sisters spend every afternoon collecting signatures. After n1 weeks, Thandiwe and Olivia compare their autograph books, counting up the number of signatures each sister has collected. Thandiwe has s1 signatures in her book, and Olivia has s2. The sisters have n2 more weeks of spring break, and they decide they want to reach goal signatures between them by the end of the break. — FIND: How many signatures do the sisters need to collect to reach their goal? → `answer = ?`
**break**  (s1: 23→22) Thandiwe and Olivia are sisters from Lakeside who love collecting signatures from politicians. During their spring break from school, the sisters spend every afternoon collecting signatures. After 6 weeks, Thandiwe and Olivia compare their autograph books, counting up the number of signatures each sister has collected. Thandiwe has 22 signatures in her book, and Olivia has 59. The sisters have 2 more weeks of spring break, and they decide they want to reach 100 signatures between them by the end of the break. How many signatures do the sisters need to collect to reach their goal?  → answer: **19** (was 18)

---

### 35. `gsms_0034` — answer: **137200**

**base**  Dr. Smith sold his cottage for $820000. He paid the registration fees that amount to 1% of the selling price and also paid an agent fee that is 3% of the selling price. If he also paid $650000 for the remaining loan amount of the cottage, how much is Dr. Smith's net proceeds from selling the cottage?
**para**  Dr. Smith sold his cottage for $820000. 1% of the selling price went on registration fees, a further 3% went on an agent fee, and $650000 went on the remaining loan amount of the cottage. What were Dr. Smith's net proceeds from selling the cottage?  <sub>(topicalization)</sub>
**formal**  GIVEN price=820000, fee1_percent=1, fee2_percent=3, loan=650000 — WHERE: Dr. Smith sold his cottage for $price. He paid the registration fees that amount to fee1_percent% of the selling price and also paid an agent fee that is fee2_percent% of the selling price. — FIND: If he also paid $loan for the remaining loan amount of the cottage, how much is Dr. Smith's net proceeds from selling the cottage? → `answer = ?`
**break**  (price: 820000→810000) Dr. Smith sold his cottage for $810000. He paid the registration fees that amount to 1% of the selling price and also paid an agent fee that is 3% of the selling price. If he also paid $650000 for the remaining loan amount of the cottage, how much is Dr. Smith's net proceeds from selling the cottage?  → answer: **127600** (was 137200)

---

### 36. `gsms_0035` — answer: **316**

**base**  Luis is raising money for a sports tournament. He has applied for help from the school, which has decided to cover a fifth of the cost of the sports tournament. How much money is Luis missing if he has ¥20 and the sports tournament costs ¥420?
**para**  Luis is raising money for a sports tournament that costs ¥420. The school he applied to for help has decided to cover a fifth of that cost, and Luis himself has ¥20. How much money is he still missing?  <sub>(sentence_split)</sub>
**formal**  GIVEN fraction=Fraction(1, 5), current=20, total=420 — WHERE: Luis is raising money for a sports tournament. He has applied for help from the school, which has decided to cover fraction of the cost of the sports tournament. — FIND: How much money is Luis missing if he has ¥current and the sports tournament costs ¥total? → `answer = ?`
**break**  (current: 20→25) Luis is raising money for a sports tournament. He has applied for help from the school, which has decided to cover a fifth of the cost of the sports tournament. How much money is Luis missing if he has ¥25 and the sports tournament costs ¥420?  → answer: **311** (was 316)

---

### 37. `gsms_0036` — answer: **752**

**base**  Ines went to the patisserie and bought various types of treats. She bought 2 dozen croissants which cost $15 per dozen, 6 dozen cookies which cost $82 per dozen, and 2 dozen scones for $115 per dozen. How much was the total cost?
**para**  At the patisserie, Ines bought various types of treats: 2 dozen croissants at $15 per dozen, 6 dozen cookies at $82 per dozen, and 2 dozen scones at $115 per dozen. What was the total cost?  <sub>(sentence_join)</sub>
**formal**  GIVEN n1=2, p1=15, n2=6, p2=82, n3=2, p3=115 — WHERE: Ines went to the patisserie and bought various types of treats. She bought n1 dozen croissants which cost $p1 per dozen, n2 dozen cookies which cost $p2 per dozen, and n3 dozen scones for $p3 per dozen. — FIND: How much was the total cost? → `answer = ?`
**break**  (n1: 2→1) Ines went to the patisserie and bought various types of treats. She bought 1 dozen croissants which cost $15 per dozen, 6 dozen cookies which cost $82 per dozen, and 2 dozen scones for $115 per dozen. How much was the total cost?  → answer: **737** (was 752)

---

### 38. `gsms_0037` — answer: **4**

**base**  A cake has 36 portions. If Patricia can eat 3 portions a day while Priya can eat twice as much, how many days will the cake last?
**para**  Patricia can eat 3 portions of a cake a day, and Priya can eat twice as much. If the cake has 36 portions, how many days will it last?  <sub>(clause_reorder)</sub>
**formal**  GIVEN n=36, x=3, mult=2 — WHERE: A cake has n portions. — FIND: If Patricia can eat x portions a day while Priya can eat mult as much, how many days will the cake last? → `answer = ?`
**break**  (x: 3→2) A cake has 36 portions. If Patricia can eat 2 portions a day while Priya can eat twice as much, how many days will the cake last?  → answer: **6** (was 4)

---

### 39. `gsms_0038` — answer: **6**

**base**  A loaf of bread at the bakery costs £4. Muffins cost £1 each. How much more do 3 loaves of bread cost than 6 muffins?
**para**  At the bakery, a loaf of bread costs £4 and a muffin costs £1. By how much does the cost of 3 loaves of bread exceed the cost of 6 muffins?  <sub>(sentence_join)</sub>
**formal**  GIVEN price1=4, price2=1, n1=3, n2=6 — WHERE: A loaf of bread at the bakery costs £price1. Muffins cost £price2 each. — FIND: How much more do n1 loaves of bread cost than n2 muffins? → `answer = ?`
**break**  (price1: 4→3) A loaf of bread at the bakery costs £3. Muffins cost £1 each. How much more do 3 loaves of bread cost than 6 muffins?  → answer: **3** (was 6)

---

### 40. `gsms_0039` — answer: **50**

**base**  A janitor has to clean a university with 120 floors. They have 6 days to get it done. It takes them 15 minutes per floor. If they work 10 hours each day, what percentage of their day, on average, is spent cleaning floors?
**para**  A university with 120 floors has to be cleaned by a janitor, who has 6 days for the work and takes 15 minutes per floor. Working 10 hours each day, what percentage of the day does the janitor spend, on average, cleaning floors?  <sub>(passive_voice)</sub>
**formal**  GIVEN n=120, d=6, t=15, h=10 — WHERE: A janitor has to clean a university with n floors. They have d days to get it done. It takes them t minutes per floor. — FIND: If they work h hours each day, what percentage of their day, on average, is spent cleaning floors? → `answer = ?`
**break**  (d: 6→5) A janitor has to clean a university with 120 floors. They have 5 days to get it done. It takes them 15 minutes per floor. If they work 10 hours each day, what percentage of their day, on average, is spent cleaning floors?  → answer: **60** (was 50)

---

### 41. `gsms_0040` — answer: **578**

**base**  Mary went to the pier for vacation. Her parents gave her $1300 to buy whatever she wanted. At the souvenir store, rock candy was on sale for "Buy 16 kilograms at $21 per kilogram, get 4 kilograms 1/4 off." She scooped up 22 kilograms. She also bought a mixed bag of sea glass for $11.25 and 13 pins that were $20.75 each. How much money does Mary have left?
**para**  On vacation at the pier, Mary was given $1300 by her parents to buy whatever she wanted. The souvenir store had rock candy on sale: "Buy 16 kilograms at $21 per kilogram, get 4 kilograms 1/4 off." She scooped up 22 kilograms, and she also bought a mixed bag of sea glass for $11.25 along with 13 pins at $20.75 each. How much money does Mary have left?  <sub>(passive_voice)</sub>
**formal**  GIVEN total=1300, n1=16, p1=21, n2=4, discount=Fraction(1, 4), n12=22, p2=11.25, n3=13, p3=20.75 — WHERE: Mary went to the pier for vacation. Her parents gave her $total to buy whatever she wanted. At the souvenir store, rock candy was on sale for "Buy n1 kilograms at $p1 per kilogram, get n2 kilograms discount off." She scooped up n12 kilograms. She also bought a mixed bag of sea glass for $p2 and n3 pins that were $p3 each. — FIND: How much money does Mary have left? → `answer = ?`
**break**  (total: 1300→1200) Mary went to the pier for vacation. Her parents gave her $1200 to buy whatever she wanted. At the souvenir store, rock candy was on sale for "Buy 16 kilograms at $21 per kilogram, get 4 kilograms 1/4 off." She scooped up 22 kilograms. She also bought a mixed bag of sea glass for $11.25 and 13 pins that were $20.75 each. How much money does Mary have left?  → answer: **478** (was 578)

---

### 42. `gsms_0041` — answer: **11**

**base**  A stack is made out of 3 purple cubes, double as many black cubes, and an unknown number of orange cubes. If there are 20 cubes in the stack in total, how many orange cubes are there?
**para**  3 purple cubes, double as many black cubes, and an unknown number of orange cubes make up a stack. How many orange cubes are there, if the stack contains 20 cubes in total?  <sub>(active_voice)</sub>
**formal**  GIVEN n1=3, mult=2, total=20 — WHERE: A stack is made out of n1 purple cubes, mult as many black cubes, and an unknown number of orange cubes. — FIND: If there are total cubes in the stack in total, how many orange cubes are there? → `answer = ?`
**break**  (n1: 3→2) A stack is made out of 2 purple cubes, double as many black cubes, and an unknown number of orange cubes. If there are 20 cubes in the stack in total, how many orange cubes are there?  → answer: **14** (was 11)

---

### 43. `gsms_0042` — answer: **368**

**base**  Wei goes to the store to buy some rice. The normal brand of rice he buys costs £20 per kilogram. He had to buy a more expensive brand that costs 30% more since his favorite brand was sold out. He decides to buy a week's worth of rice and he uses 2 kilogram of rice per day. He also decided to buy himself a bagel for £4. How much did everything cost?
**para**  Wei went to the store for rice. His usual brand, at £20 per kilogram, was sold out, so he had to take a brand costing 30% more. He bought a week's worth, using 2 kilogram of rice per day, and he treated himself to a bagel for £4 as well. What did everything cost?  <sub>(clause_reorder)</sub>
**formal**  GIVEN price=20, percent=30, usage=2, extra_price=4 — WHERE: Wei goes to the store to buy some rice. The normal brand of rice he buys costs £price per kilogram. He had to buy a more expensive brand that costs percent% more since his favorite brand was sold out. He decides to buy a week's worth of rice and he uses usage kilogram of rice per day. He also decided to buy himself a bagel for £extra_price. — FIND: How much did everything cost? → `answer = ?`
**break**  (price: 20→10) Wei goes to the store to buy some rice. The normal brand of rice he buys costs £10 per kilogram. He had to buy a more expensive brand that costs 30% more since his favorite brand was sold out. He decides to buy a week's worth of rice and he uses 2 kilogram of rice per day. He also decided to buy himself a bagel for £4. How much did everything cost?  → answer: **186** (was 368)

---

### 44. `gsms_0043` — answer: **196**

**base**  Linda went to their greenhouse to pick some cherries and found 1/3 as many caterpillars as grasshoppers in the patch. If there were 147 grasshoppers, calculate the total number of insects in the patch.
**para**  While picking cherries at their greenhouse, Linda found that the patch held 1/3 as many caterpillars as grasshoppers. There were 147 grasshoppers. What was the total number of insects in the patch?  <sub>(imperative_to_wh)</sub>
**formal**  GIVEN frac=Fraction(1, 3), n=147 — WHERE: Linda went to their greenhouse to pick some cherries and found frac as many caterpillars as grasshoppers in the patch. — FIND: If there were n grasshoppers, calculate the total number of insects in the patch. → `answer = ?`
**break**  (n: 147→144) Linda went to their greenhouse to pick some cherries and found 1/3 as many caterpillars as grasshoppers in the patch. If there were 144 grasshoppers, calculate the total number of insects in the patch.  → answer: **192** (was 196)

---

### 45. `gsms_0044` — answer: **30**

**base**  There are 10 girls in the yard. If there are twice the number of boys in the yard, how many kids are in the yard?
**para**  The yard holds 10 girls, and it holds twice the number of boys. How many kids are in the yard?  <sub>(sentence_join)</sub>
**formal**  GIVEN n=10, mult=2 — WHERE: There are n girls in the yard. — FIND: If there are mult the number of boys in the yard, how many kids are in the yard? → `answer = ?`
**break**  (n: 10→11) There are 11 girls in the yard. If there are twice the number of boys in the yard, how many kids are in the yard?  → answer: **33** (was 30)

---

### 46. `gsms_0045` — answer: **8**

**base**  Amara has a flower bed that is 204 meters long. Amara wants to fill her flower bed with plants. Amara's flowers grow 7 meters wide so she needs to leave 12.75 meters between every plant. Amara already owns 15 flowers. Each flowering plant costs €8 at the store, how much money will Amara spend at the store to fill up her flower bed?
**para**  Amara wants to fill a flower bed 204 meters long with plants. Because her flowers grow 7 meters wide, she has to leave 12.75 meters between every plant. She already owns 15 flowers, and each flowering plant costs €8 at the store. How much money will Amara spend at the store to fill up her flower bed?  <sub>(sentence_join)</sub>
**formal**  GIVEN length=204, plant_width=7, space=12.75, owned=15, cost=8 — WHERE: Amara has a flower bed that is length meters long. Amara wants to fill her flower bed with plants. Amara's flowers grow plant_width meters wide so she needs to leave space meters between every plant. Amara already owns owned flowers. — FIND: Each flowering plant costs €cost at the store, how much money will Amara spend at the store to fill up her flower bed? → `answer = ?`
**break**  (owned: 15→14) Amara has a flower bed that is 204 meters long. Amara wants to fill her flower bed with plants. Amara's flowers grow 7 meters wide so she needs to leave 12.75 meters between every plant. Amara already owns 14 flowers. Each flowering plant costs €8 at the store, how much money will Amara spend at the store to fill up her flower bed?  → answer: **16** (was 8)

---

### 47. `gsms_0046` — answer: **6**

**base**  Oakwood High School is holding graduation in their Auditorium this year which has space for 6400 people. After accounting for 900 seats for graduates and 100 seats for faculty attending, how many tickets would each graduate receive to give to their friends and family if the tickets are split equally?
**para**  Graduation at Oakwood High School this year is being held in their Auditorium, which has space for 6400 people. Once 900 seats for graduates and 100 seats for faculty attending have been accounted for, the tickets that remain are split equally among the graduates. How many tickets would each graduate receive to give to their friends and family?  <sub>(sentence_split)</sub>
**formal**  GIVEN total=6400, graduates=900, faculty=100 — WHERE: Oakwood High School is holding graduation in their Auditorium this year which has space for total people. — FIND: After accounting for graduates seats for graduates and faculty seats for faculty attending, how many tickets would each graduate receive to give to their friends and family if the tickets are split equally? → `answer = ?`
**break**  (graduates: 900→700) Oakwood High School is holding graduation in their Auditorium this year which has space for 6400 people. After accounting for 700 seats for graduates and 100 seats for faculty attending, how many tickets would each graduate receive to give to their friends and family if the tickets are split equally?  → answer: **8** (was 6)

---

### 48. `gsms_0047` — answer: **5**

**base**  Hiroshi hid 6 dozen eggs in the yard for the Easter egg hunt. Noah finds 7 eggs. Aisha finds four times as many as Noah. Luis finds 4 less than Aisha, and Elias finds a third as many as Luis. How many eggs are still hidden in the yard?
**para**  6 dozen eggs were hidden in the yard by Hiroshi for the Easter egg hunt. Noah finds 7 eggs; Aisha finds four times as many as Noah; Luis finds 4 less than Aisha; and Elias finds a third as many as Luis. How many eggs are still hidden in the yard?  <sub>(passive_voice)</sub>
**formal**  GIVEN n=6, x=7, mult1=4, y=4, frac=Fraction(1, 3) — WHERE: Hiroshi hid n dozen eggs in the yard for the Easter egg hunt. Noah finds x eggs. Aisha finds mult1 as many as Noah. Luis finds y less than Aisha, and Elias finds frac as many as Luis. — FIND: How many eggs are still hidden in the yard? → `answer = ?`
**break**  (n: 6→7) Hiroshi hid 7 dozen eggs in the yard for the Easter egg hunt. Noah finds 7 eggs. Aisha finds four times as many as Noah. Luis finds 4 less than Aisha, and Elias finds a third as many as Luis. How many eggs are still hidden in the yard?  → answer: **17** (was 5)

---

### 49. `gsms_0048` — answer: **33**

**base**  Zara can peel 8 mushrooms a minute and saute 40 mushrooms in 6 minutes. How long will it take her to peel and saute 120 mushrooms?
**para**  Peeling mushrooms, Zara manages 8 a minute; sauteing them, she manages 40 in 6 minutes. How long will it take her to peel and saute 120 mushrooms?  <sub>(nominalization)</sub>
**formal**  GIVEN n1=8, n2=40, t=6, total=120 — WHERE: Zara can peel n1 mushrooms a minute and saute n2 mushrooms in t minutes. — FIND: How long will it take her to peel and saute total mushrooms? → `answer = ?`
**break**  (n1: 8→6) Zara can peel 6 mushrooms a minute and saute 40 mushrooms in 6 minutes. How long will it take her to peel and saute 120 mushrooms?  → answer: **38** (was 33)

---

### 50. `gsms_0049` — answer: **8820**

**base**  Ahmed collects garbage from 5 different apartment complexes. The first 4 have 450 apartments each and the last one is 45% bigger than the other 4 combined. Ahmed collects garbage 4 times a week from each place and he gets paid $0.5 per collection for each apartment. How much money does he make in a week?
**para**  Garbage is collected by Ahmed from 5 different apartment complexes. Each of the first 4 has 450 apartments, and the last one is 45% bigger than the other 4 combined. He collects 4 times a week from each place, and each collection pays him $0.5 for each apartment. How much money does he make in a week?  <sub>(passive_voice)</sub>
**formal**  GIVEN n=5, n_first=4, a=450, p=45, f=4, r=0.5 — WHERE: Ahmed collects garbage from n different apartment complexes. The first n_first have a apartments each and the last one is p% bigger than the other n_first combined. Ahmed collects garbage f times a week from each place and he gets paid $r per collection for each apartment. — FIND: How much money does he make in a week? → `answer = ?`
**break**  (a: 450→400) Ahmed collects garbage from 5 different apartment complexes. The first 4 have 400 apartments each and the last one is 45% bigger than the other 4 combined. Ahmed collects garbage 4 times a week from each place and he gets paid $0.5 per collection for each apartment. How much money does he make in a week?  → answer: **7840** (was 8820)

---

### 51. `gsms_0050` — answer: **120**

**base**  The area of Ravi's rectangular study is 675 square feet. If the length of his  room is 5 yards, what is the perimeter of the room in feet?
**para**  Ravi's rectangular study has an area of 675 square feet. Given that his room is 5 yards long, what is the perimeter of the room in feet?  <sub>(fronted_conditional)</sub>
**formal**  GIVEN area=675, length=5 — WHERE: The area of Ravi's rectangular study is area square feet. — FIND: If the length of his  room is length yards, what is the perimeter of the room in feet? → `answer = ?`
**break**  (area: 675→645) The area of Ravi's rectangular study is 645 square feet. If the length of his  room is 5 yards, what is the perimeter of the room in feet?  → answer: **116** (was 120)

---

### 52. `gsms_0051` — answer: **269**

**base**  David has a herb stand at the Community market. He sells three kinds of herbs: Tulips, sunflowers and lilies. He usually sells tulips for $2.58 per basket, sunflowers for $2.53 per basket and lilies for $4.85 per basket. David has no change today, so he has decided to round all his prices to the nearest dollar. If David sells 5 baskets of tulips, 18 baskets of sunflowers and 40 baskets of lilies, how much will he make?
**para**  David runs a herb stand at the community market, selling three kinds of herbs: Tulips, sunflowers and lilies. His usual prices are $2.58 per basket for tulips, $2.53 per basket for sunflowers and $4.85 per basket for lilies, but today he has no change, so he has decided to round all his prices to the nearest dollar. How much will he make if he sells 5 baskets of tulips, 18 baskets of sunflowers and 40 baskets of lilies?  <sub>(sentence_join)</sub>
**formal**  GIVEN p1=2.58, p2=2.53, p3=4.85, n1=5, n2=18, n3=40 — WHERE: David has a herb stand at the Community market. He sells three kinds of herbs: Tulips, sunflowers and lilies. He usually sells tulips for $p1 per basket, sunflowers for $p2 per basket and lilies for $p3 per basket. David has no change today, so he has decided to round all his prices to the nearest dollar. — FIND: If David sells n1 baskets of tulips, n2 baskets of sunflowers and n3 baskets of lilies, how much will he make? → `answer = ?`
**break**  (n1: 5→6) David has a herb stand at the Community market. He sells three kinds of herbs: Tulips, sunflowers and lilies. He usually sells tulips for $2.58 per basket, sunflowers for $2.53 per basket and lilies for $4.85 per basket. David has no change today, so he has decided to round all his prices to the nearest dollar. If David sells 6 baskets of tulips, 18 baskets of sunflowers and 40 baskets of lilies, how much will he make?  → answer: **272** (was 269)

---

### 53. `gsms_0052` — answer: **87**

**base**  Jamal has 5 ¥20 bills. He buys 3 toy cars for ¥1 each. He also buys 2 packs of candy bars for ¥5 each. How much money does he have left?
**para**  Jamal starts with 5 ¥20 bills, spends ¥1 each on 3 toy cars, and spends ¥5 each on 2 packs of candy bars. How much money does he have left?  <sub>(sentence_join)</sub>
**formal**  GIVEN n=5, bill=20, m=3, p1=1, k=2, p2=5 — WHERE: Jamal has n ¥bill bills. He buys m toy cars for ¥p1 each. He also buys k packs of candy bars for ¥p2 each. — FIND: How much money does he have left? → `answer = ?`
**break**  (n: 5→4) Jamal has 4 ¥20 bills. He buys 3 toy cars for ¥1 each. He also buys 2 packs of candy bars for ¥5 each. How much money does he have left?  → answer: **67** (was 87)

---

### 54. `gsms_0053` — answer: **474**

**base**  Marcus scored 294 points in one game of tennis. Sofia scored 33 points more than half as many as Marcus. How many points did Marcus and Sofia have in total?
**para**  In one game of tennis, Marcus scored 294 points, and Sofia scored 33 points more than half as many as Marcus. How many points did the two of them have in total?  <sub>(sentence_join)</sub>
**formal**  GIVEN n1=294, n2=33, frac=Fraction(1, 2) — WHERE: Marcus scored n1 points in one game of tennis. Sofia scored n2 points more than frac as many as Marcus. — FIND: How many points did Marcus and Sofia have in total? → `answer = ?`
**break**  (n1: 294→292) Marcus scored 292 points in one game of tennis. Sofia scored 33 points more than half as many as Marcus. How many points did Marcus and Sofia have in total?  → answer: **471** (was 474)

---

### 55. `gsms_0054` — answer: **27**

**base**  Leila sliced a whole peach into 28 pieces. She ate 4 slices, her brother ate 6 more than her, and her friend ate 3 more than her brother. How many slices of peach did they all eat?
**para**  A whole peach was sliced into 28 pieces by Leila. She ate 4 slices; her brother ate 6 more than she did; and her friend ate 3 more than her brother. How many slices of peach did they all eat?  <sub>(passive_voice)</sub>
**formal**  GIVEN total=28, n1=4, n2=6, n3=3 — WHERE: Leila sliced a whole peach into total pieces. She ate n1 slices, her brother ate n2 more than her, and her friend ate n3 more than her brother. — FIND: How many slices of peach did they all eat? → `answer = ?`
**break**  (n1: 4→3) Leila sliced a whole peach into 28 pieces. She ate 3 slices, her brother ate 6 more than her, and her friend ate 3 more than her brother. How many slices of peach did they all eat?  → answer: **24** (was 27)

---

### 56. `gsms_0055` — answer: **336**

**base**  Yuki is learning to write and decides to keep re-writing the farsi alphabet (with 32 letters) until she knows it. She writes it in full five times, writes a quarter of it once, then re-writes everything she has already written. How many letters has Yuki written in total?
**para**  Learning to write, Yuki decides to keep re-writing the farsi alphabet (with 32 letters) until she knows it. After writing it out in full five times and writing a quarter of it once, she re-writes everything she has already written. In total, how many letters has Yuki written?  <sub>(nominalization)</sub>
**formal**  GIVEN alphabets=32, n1=5, frac=Fraction(1, 4) — WHERE: Yuki is learning to write and decides to keep re-writing the alphabets until she knows it. She writes it in full n1, writes frac of it once, then re-writes everything she has already written. — FIND: How many letters has Yuki written in total? → `answer = ?`
**break**  (n1: five times→four times) Yuki is learning to write and decides to keep re-writing the farsi alphabet (with 32 letters) until she knows it. She writes it in full four times, writes a quarter of it once, then re-writes everything she has already written. How many letters has Yuki written in total?  → answer: **272** (was 336)

---

### 57. `gsms_0056` — answer: **48**

**base**  A bottle of tea is 6 ozs of tea. Ali drinks 4 bottles of tea. If tea has 2 calories per oz how many calories did he consume?
**para**  A bottle of tea holds 6 ozs. Ali drinks 4 bottles of it. Given 2 calories per oz of tea, how many calories did he consume?  <sub>(fronted_conditional)</sub>
**formal**  GIVEN vol=6, num=4, cal=2 — WHERE: A bottle of tea is vol ozs of tea. Ali drinks num bottles of tea. — FIND: If tea has cal calories per oz how many calories did he consume? → `answer = ?`
**break**  (vol: 6→7) A bottle of tea is 7 ozs of tea. Ali drinks 4 bottles of tea. If tea has 2 calories per oz how many calories did he consume?  → answer: **56** (was 48)

---

### 58. `gsms_0057` — answer: **23**

**base**  Andre has 15 tubes of red paint and 9 tubes of yellow paint. David has 1/3 as many tubes of red paint as Andre, and double as many tubes of yellow paint as Andre. How many tubes of paint does David have?
**para**  David has 1/3 as many tubes of red paint as Andre and double as many tubes of yellow paint as Andre, who has 15 tubes of red paint and 9 tubes of yellow paint. How many tubes of paint does David have?  <sub>(relative_clause)</sub>
**formal**  GIVEN n1=15, n2=9, frac1=Fraction(1, 3), mult1=2 — WHERE: Andre has n1 tubes of red paint and n2 tubes of yellow paint. David has frac1 as many tubes of red paint as Andre, and mult1 as many tubes of yellow paint as Andre. — FIND: How many tubes of paint does David have? → `answer = ?`
**break**  (n1: 15→12) Andre has 12 tubes of red paint and 9 tubes of yellow paint. David has 1/3 as many tubes of red paint as Andre, and double as many tubes of yellow paint as Andre. How many tubes of paint does David have?  → answer: **22** (was 23)

---

### 59. `gsms_0058` — answer: **9802**

**base**  One goblin can mine 5 pounds of ore per day with bare hands. He can mine double as much with a silver pickaxe and 30% more with a mithril pickaxe than with a silver pickaxe. How many pounds of ore can 26 goblins with mithril pickaxes mine in a month with 29 days?
**para**  With bare hands, one goblin can mine 5 pounds of ore per day. A silver pickaxe lets him mine double as much, and a mithril pickaxe lets him mine 30% more than a silver pickaxe does. In a month of 29 days, how many pounds of ore can 26 goblins with mithril pickaxes mine?  <sub>(topicalization)</sub>
**formal**  GIVEN base=5, mult1=2, mult2=30, n=26, days=29 — WHERE: One goblin can mine base pounds of ore per day with bare hands. He can mine mult1 as much with a silver pickaxe and mult2% more with a mithril pickaxe than with a silver pickaxe. — FIND: How many pounds of ore can n goblins with mithril pickaxes mine in a month with days days? → `answer = ?`
**break**  (n: 26→25) One goblin can mine 5 pounds of ore per day with bare hands. He can mine double as much with a silver pickaxe and 30% more with a mithril pickaxe than with a silver pickaxe. How many pounds of ore can 25 goblins with mithril pickaxes mine in a month with 29 days?  → answer: **9425** (was 9802)

---

### 60. `gsms_0059` — answer: **17**

**base**  I have 20 liters of apple drink that are 1/3 water and I wish to add it to 24 liters of peach drink that is half water. But as I pour it, I spill 5 liters of the apple drink. How much water is in the remaining 39 liters?
**para**  20 liters of apple drink, 1/3 of which is water, are to be added to 24 liters of peach drink, half of which is water. While pouring, I spill 5 liters of the apple drink. How much water is in the remaining 39 liters?  <sub>(passive_voice)</sub>
**formal**  GIVEN n1=20, frac1=Fraction(1, 3), n2=24, frac2=Fraction(1, 2), spill=5, total=39 — WHERE: I have n1 liters of apple drink that are frac1 water and I wish to add it to n2 liters of peach drink that is frac2 water. But as I pour it, I spill spill liters of the apple drink. — FIND: How much water is in the remaining total liters? → `answer = ?`
**break**  (frac1: 1/3→1/5) I have 20 liters of apple drink that are 1/5 water and I wish to add it to 24 liters of peach drink that is half water. But as I pour it, I spill 5 liters of the apple drink. How much water is in the remaining 39 liters?  → answer: **15** (was 17)

---

### 61. `gsms_0060` — answer: **4052**

**base**  Kai works a 12-hour shift each day, 5 days a week. He earns £12 per hour and gets a £293 bonus each week if the company performs well. How much money did Kai make in September if the company performed very well for the whole month?
**para**  Kai earns £12 per hour for a 12-hour shift, which he works 5 days a week, and in each week the company performs well he receives a £293 bonus. If the company performed very well for the whole of September, how much money did Kai make that month?  <sub>(relative_clause)</sub>
**formal**  GIVEN h=12, d=5, r=12, b=293 — WHERE: Kai works a h-hour shift each day, d days a week. He earns £r per hour and gets a £b bonus each week if the company performs well. — FIND: How much money did Kai make in September if the company performed very well for the whole month? → `answer = ?`
**break**  (h: 12→11) Kai works a 11-hour shift each day, 5 days a week. He earns £12 per hour and gets a £293 bonus each week if the company performs well. How much money did Kai make in September if the company performed very well for the whole month?  → answer: **3812** (was 4052)

---

### 62. `gsms_0061` — answer: **49**

**base**  Mateo has a bag of coins with 70 inside. He tripped over root while carrying it and dropped half of them. He scrambled to search for them but only came up with 17. When he went back home, he inspected the coins further. Three of them he picked up weren't coins, but actually beads so he got rid of it. How many coins did Mateo end up with?
**para**  Mateo was carrying a bag with 70 coins inside when he tripped over root and dropped half of them, and searching around he came up with only 17. He inspected the coins more closely once he was home. Three of the ones he had picked up turned out not to be coins at all but beads, so he got rid of them. How many coins did Mateo end up with?  <sub>(sentence_join)</sub>
**formal**  GIVEN n=70, frac=Fraction(1, 2), k=17, fake_num=3 — WHERE: Mateo has a bag of coins with n inside. He tripped over root while carrying it and dropped frac of them. He scrambled to search for them but only came up with k. When he went back home, he inspected the coins further. fake_num of them he picked up weren't coins, but actually beads so he got rid of it. — FIND: How many coins did Mateo end up with? → `answer = ?`
**break**  (n: 70→72) Mateo has a bag of coins with 72 inside. He tripped over root while carrying it and dropped half of them. He scrambled to search for them but only came up with 17. When he went back home, he inspected the coins further. Three of them he picked up weren't coins, but actually beads so he got rid of it. How many coins did Mateo end up with?  → answer: **50** (was 49)

---

### 63. `gsms_0062` — answer: **102**

**base**  In a set of magician's cards, there are 30 red cards, and 70% more blue cards. Green cards are as many as the sum of red and blue cards. How many cards of all mentioned colors are there?
**para**  A set of magician's cards contains 30 red cards and 70% more blue cards than that, with as many green cards as the red and blue cards put together. How many cards of all mentioned colors are there?  <sub>(sentence_join)</sub>
**formal**  GIVEN n1=30, p=70 — WHERE: In a set of magician's cards, there are n1 red cards, and p% more blue cards. Green cards are as many as the sum of red and blue cards. — FIND: How many cards of all mentioned colors are there? → `answer = ?`
**break**  (n1: 30→20) In a set of magician's cards, there are 20 red cards, and 70% more blue cards. Green cards are as many as the sum of red and blue cards. How many cards of all mentioned colors are there?  → answer: **68** (was 102)

---

### 64. `gsms_0063` — answer: **22**

**base**  A kid has 14 toys. His friend has 6 fewer toys than he has. How many toys do they have together?
**para**  A kid owns 14 toys. His friend owns 6 fewer than that. Together, how many toys do they have?  <sub>(sentence_split)</sub>
**formal**  GIVEN n=14, k=6 — WHERE: A kid has n toys. His friend has k fewer toys than he has. — FIND: How many toys do they have together? → `answer = ?`
**break**  (n: 14→13) A kid has 13 toys. His friend has 6 fewer toys than he has. How many toys do they have together?  → answer: **20** (was 22)

---

### 65. `gsms_0064` — answer: **16**

**base**  Jennifer has 11 jars of icing in her storage room. Each jar of icing can decorate 16 cupcakes. Jennifer wants to bake enough cupcakes to use up all of her icing. If each tray holds 11 cupcakes, how many trays worth of cupcakes should she bake?
**para**  In her storage room, Jennifer has 11 jars of icing, each of which can decorate 16 cupcakes, and she wants to bake enough cupcakes to use all of it up. If a tray holds 11 cupcakes, how many trays worth of cupcakes should she bake?  <sub>(relative_clause)</sub>
**formal**  GIVEN n=11, m=16, k=11 — WHERE: Jennifer has n jars of icing in her storage room. Each jar of icing can decorate m cupcakes. Jennifer wants to bake enough cupcakes to use up all of her icing. — FIND: If each tray holds k cupcakes, how many trays worth of cupcakes should she bake? → `answer = ?`
**break**  (m: 16→15) Jennifer has 11 jars of icing in her storage room. Each jar of icing can decorate 15 cupcakes. Jennifer wants to bake enough cupcakes to use up all of her icing. If each tray holds 11 cupcakes, how many trays worth of cupcakes should she bake?  → answer: **15** (was 16)

---

### 66. `gsms_0065` — answer: **289**

**base**  Olivia is trying to decide whether to do her taxes herself or hire one tax consultant. If she does it herself, she'll be able to do 9 fewer hours of consulting work, losing £42/hour in missed income. The tax consultant charges £89. How much more money will she have if she hires the tax consultant?
**para**  Olivia must decide between doing her taxes herself and hiring one tax consultant. Doing it herself would cost her 9 hours of consulting work and the £42/hour of income those hours would have brought in, while the tax consultant charges £89. How much more money will she have if she hires the tax consultant?  <sub>(nominalization)</sub>
**formal**  GIVEN n=9, rate=42, fee=89 — WHERE: Olivia is trying to decide whether to do her taxes herself or hire one tax consultant. If she does it herself, she'll be able to do n fewer hours of consulting work, losing £rate/hour in missed income. The tax consultant charges £fee. — FIND: How much more money will she have if she hires the tax consultant? → `answer = ?`
**break**  (n: 9→8) Olivia is trying to decide whether to do her taxes herself or hire one tax consultant. If she does it herself, she'll be able to do 8 fewer hours of consulting work, losing £42/hour in missed income. The tax consultant charges £89. How much more money will she have if she hires the tax consultant?  → answer: **247** (was 289)

---

### 67. `gsms_0066` — answer: **62**

**base**  Amara has 81 light orange spools of thread, 136 dark orange spools of thread, 35 light yellow spools of thread, and 98 dark yellow spools of thread. What percent of the spools are orange?
**para**  The spools of thread Amara has are 81 light orange, 136 dark orange, 35 light yellow, and 98 dark yellow. What percent of the spools are orange?  <sub>(topicalization)</sub>
**formal**  GIVEN n1=81, n2=136, n3=35, n4=98 — WHERE: Amara has n1 light orange spools of thread, n2 dark orange spools of thread, n3 light yellow spools of thread, and n4 dark yellow spools of thread. — FIND: What percent of the spools are orange? → `answer = ?`
**break**  (n1: 81→80) Amara has 80 light orange spools of thread, 136 dark orange spools of thread, 35 light yellow spools of thread, and 98 dark yellow spools of thread. What percent of the spools are orange?  → answer: **61** (was 62)

---

### 68. `gsms_0067` — answer: **26**

**base**  Kai's van breaks down. The van weighs 2350 pounds and he has groceries in it weighing 100 pounds. He also has his 2 young children who weigh 75 pounds each in it. If the force to move the van is 1% of the weight, how much force does he need to push the van?
**para**  Kai's van breaks down. Inside it are groceries weighing 100 pounds and his 2 young children, who weigh 75 pounds each; the van itself weighs 2350 pounds. If the force to move the van is 1% of the weight, how much force does he need to push the van?  <sub>(clause_reorder)</sub>
**formal**  GIVEN weight_vehicle=2350, weight_item=100, num_passengers=2, weight_passenger=75, force_percent=1 — WHERE: Kai's van breaks down. The van weighs weight_vehicle pounds and he has groceries in it weighing weight_item pounds. He also has his num_passengers young children who weigh weight_passenger pounds each in it. — FIND: If the force to move the van is force_percent% of the weight, how much force does he need to push the van? → `answer = ?`
**break**  (weight_vehicle: 2350→2250) Kai's van breaks down. The van weighs 2250 pounds and he has groceries in it weighing 100 pounds. He also has his 2 young children who weigh 75 pounds each in it. If the force to move the van is 1% of the weight, how much force does he need to push the van?  → answer: **25** (was 26)

---

### 69. `gsms_0068` — answer: **268**

**base**  It takes Sofia 27 minutes to finish a word puzzle and 17 minutes to finish a jigsaw puzzle. Over the vacation she solved 3 word puzzles and 11 jigsaw puzzles. How much time did she spend playing these games?
**para**  A word puzzle takes Sofia 27 minutes to finish, and a jigsaw puzzle takes her 17 minutes. She solved 3 word puzzles and 11 jigsaw puzzles over the vacation. How much time did she spend playing these games?  <sub>(topicalization)</sub>
**formal**  GIVEN t1=27, t2=17, n1=3, n2=11 — WHERE: It takes Sofia t1 minutes to finish a word puzzle and t2 minutes to finish a jigsaw puzzle. Over the vacation she solved n1 word puzzles and n2 jigsaw puzzles. — FIND: How much time did she spend playing these games? → `answer = ?`
**break**  (t1: 27→26) It takes Sofia 26 minutes to finish a word puzzle and 17 minutes to finish a jigsaw puzzle. Over the vacation she solved 3 word puzzles and 11 jigsaw puzzles. How much time did she spend playing these games?  → answer: **265** (was 268)

---

### 70. `gsms_0069` — answer: **295**

**base**  On Monday, Diego wants to exercise for five times the amount of time he did on Sunday and Saturday combined. On Saturday he exercised for 11 minutes. On Sunday he exercised for 48 minutes. How many minutes does he have to exercise on Monday to reach his goal?
**para**  Diego exercised for 11 minutes on Saturday and for 48 minutes on Sunday. His goal for Monday is to exercise five times the amount of time he did on those two days combined. How many minutes does he have to exercise on Monday to reach his goal?  <sub>(clause_reorder)</sub>
**formal**  GIVEN mult=5, t1=11, t2=48 — WHERE: On Monday, Diego wants to exercise for mult the amount of time he did on Sunday and Saturday combined. On Saturday he exercised for t1 minutes. On Sunday he exercised for t2 minutes. — FIND: How many minutes does he have to exercise on Monday to reach his goal? → `answer = ?`
**break**  (t1: 11→10) On Monday, Diego wants to exercise for five times the amount of time he did on Sunday and Saturday combined. On Saturday he exercised for 10 minutes. On Sunday he exercised for 48 minutes. How many minutes does he have to exercise on Monday to reach his goal?  → answer: **290** (was 295)

---

### 71. `gsms_0070` — answer: **11520**

**base**  A landscaper works for 4 weeks every month and for 6 days every week. If he gets paid ¥40 every day, how much does he earn if he works for a year?
**para**  Every month a landscaper works 4 weeks, every week he works 6 days, and every day he is paid ¥40. How much does he earn if he works for a year?  <sub>(fronted_adverbial)</sub>
**formal**  GIVEN w=4, d=6, p=40 — WHERE: A landscaper works for w weeks every month and for d days every week. — FIND: If he gets paid ¥p every day, how much does he earn if he works for a year? → `answer = ?`
**break**  (w: 4→3) A landscaper works for 3 weeks every month and for 6 days every week. If he gets paid ¥40 every day, how much does he earn if he works for a year?  → answer: **8640** (was 11520)

---

### 72. `gsms_0071` — answer: **277**

**base**  Tomas is feeding his livestock feed. Each turkey needs 3 kg, and each cow needs 1 kg less than twice the amount each turkey needs. If there are 19 turkeys and 44 cows, how many kg of feed does Tomas need?
**para**  Tomas has 19 turkeys and 44 cows to feed with feed. A turkey needs 3 kg each, while a cow needs 1 kg less than twice what a turkey needs. How many kg of feed does Tomas need?  <sub>(clause_reorder)</sub>
**formal**  GIVEN n1=3, n2=1, k1=19, k2=44 — WHERE: Tomas is feeding his livestock feed. Each turkey needs n1 kg, and each cow needs n2 kg less than twice the amount each turkey needs. — FIND: If there are k1 turkeys and k2 cows, how many kg of feed does Tomas need? → `answer = ?`
**break**  (n1: 3→4) Tomas is feeding his livestock feed. Each turkey needs 4 kg, and each cow needs 1 kg less than twice the amount each turkey needs. If there are 19 turkeys and 44 cows, how many kg of feed does Tomas need?  → answer: **384** (was 277)

---

### 73. `gsms_0072` — answer: **32**

**base**  Sanjay was a pen pal with 10 people. He stopped being penpals with 4 of them. They each send 4 letters a week that are 8 pages long. He responds in kind. He can write a page every 10 minutes. How many hours does he spend writing a week?
**para**  Sanjay was a pen pal with 10 people, 4 of whom he stopped writing to. The rest each send him 4 letters a week that are 8 pages long, and he responds in kind, writing a page every 10 minutes. How many hours does he spend writing a week?  <sub>(relative_clause)</sub>
**formal**  GIVEN n1=10, n2=4, l=4, p=8, t=10 — WHERE: Sanjay was a pen pal with n1 people. He stopped being penpals with n2 of them. They each send l letters a week that are p pages long. He responds in kind. He can write a page every t minutes. — FIND: How many hours does he spend writing a week? → `answer = ?`
**break**  (n1: 10→13) Sanjay was a pen pal with 13 people. He stopped being penpals with 4 of them. They each send 4 letters a week that are 8 pages long. He responds in kind. He can write a page every 10 minutes. How many hours does he spend writing a week?  → answer: **48** (was 32)

---

### 74. `gsms_0073` — answer: **33**

**base**  Comet Encke orbits the sun every 55 years. Kai's dad saw the Comet when he was 22 years old. Kai saw the comet a second time when he was four times the age his dad was when he saw the Comet. How old was Kai when he saw the Comet for the first time?
**para**  Every 55 years, Comet Encke orbits the sun. Kai's dad was 22 years old when he saw the Comet. When Kai saw the comet a second time, he was four times the age his dad had been at that sighting. How old was Kai when he saw the Comet for the first time?  <sub>(fronted_adverbial)</sub>
**formal**  GIVEN orbit_period=55, relative_age=22, multiple=4 — WHERE: Comet Encke orbits the sun every orbit_period years. Kai's dad saw the Comet when he was relative_age years old. Kai saw the comet a second time when he was multiple the age his dad was when he saw the Comet. — FIND: How old was Kai when he saw the Comet for the first time? → `answer = ?`
**break**  (orbit_period: 55→50) Comet Encke orbits the sun every 50 years. Kai's dad saw the Comet when he was 22 years old. Kai saw the comet a second time when he was four times the age his dad was when he saw the Comet. How old was Kai when he saw the Comet for the first time?  → answer: **38** (was 33)

---

### 75. `gsms_0074` — answer: **141**

**base**  There are 180 fourth-graders at Oak Ridge School. 35 of them are girls. On Friday, 8 fourth-grade girls and 4 fourth-grade boys were absent. How many fourth grade boys were at Oak Ridge School on Friday?
**para**  Oak Ridge School has 180 fourth-graders, 35 of them girls. 8 fourth-grade girls and 4 fourth-grade boys were absent on Friday. How many fourth grade boys were at Oak Ridge School on Friday?  <sub>(existential_to_possessive)</sub>
**formal**  GIVEN total=180, num_girls=35, absent_girls=8, absent_boys=4 — WHERE: There are total fourth-graders at Oak Ridge School. num_girls of them are girls. On Friday, absent_girls fourth-grade girls and absent_boys fourth-grade boys were absent. — FIND: How many fourth grade boys were at Oak Ridge School on Friday? → `answer = ?`
**break**  (total: 180→181) There are 181 fourth-graders at Oak Ridge School. 35 of them are girls. On Friday, 8 fourth-grade girls and 4 fourth-grade boys were absent. How many fourth grade boys were at Oak Ridge School on Friday?  → answer: **142** (was 141)

---

### 76. `gsms_0075` — answer: **378**

**base**  There are currently 5 green balls, 24 pink balls, and 41 yellow balls in the warehouse. Green balls cost $12, yellow balls cost $6, and Pink balls cost $3. How much will the warehouse have received after all the balls are sold?
**para**  The warehouse currently holds 5 green balls, 24 pink balls, and 41 yellow balls. A green ball costs $12, a yellow ball costs $6, and a pink ball costs $3. How much will the warehouse have received once all the balls are sold?  <sub>(lexical_substitution)</sub>
**formal**  GIVEN n1=5, n2=24, n3=41, p1=12, p3=6, p2=3 — WHERE: There are currently n1 green balls, n2 pink balls, and n3 yellow balls in the warehouse. Green balls cost $p1, yellow balls cost $p3, and Pink balls cost $p2. — FIND: How much will the warehouse have received after all the balls are sold? → `answer = ?`
**break**  (n1: 5→4) There are currently 4 green balls, 24 pink balls, and 41 yellow balls in the warehouse. Green balls cost $12, yellow balls cost $6, and Pink balls cost $3. How much will the warehouse have received after all the balls are sold?  → answer: **366** (was 378)

---

### 77. `gsms_0076` — answer: **333**

**base**  Ravi caught 3 steelheads last Thursday, the first steelhead he caught weighs 78 pounds, the second steelhead he caught weighs 43 pounds, and the last steelhead he caught weighs 27 pounds. If a pound of steelhead costs ¥2.25, how much will he earn after selling all the steelheads to the market?
**para**  Last Thursday, Ravi caught 3 steelheads, the first weighing 78 pounds, the second 43 pounds, and the last 27 pounds. At ¥2.25 per pound of steelhead, how much will he earn after selling all the steelheads to the market?  <sub>(fronted_adverbial)</sub>
**formal**  GIVEN n=3, w1=78, w2=43, w3=27, price=2.25 — WHERE: Ravi caught n steelheads last Thursday, the first steelhead he caught weighs w1 pounds, the second steelhead he caught weighs w2 pounds, and the last steelhead he caught weighs w3 pounds. — FIND: If a pound of steelhead costs ¥price, how much will he earn after selling all the steelheads to the market? → `answer = ?`
**break**  (n: 3→7) Ravi caught 7 steelheads last Thursday, the first steelhead he caught weighs 78 pounds, the second steelhead he caught weighs 43 pounds, and the last steelhead he caught weighs 27 pounds. If a pound of steelhead costs ¥2.25, how much will he earn after selling all the steelheads to the market?  → answer: **576** (was 333)

---

### 78. `gsms_0077` — answer: **63**

**base**  Jennifer is required to submit a 84-page report. She already finished writing a quarter of the report. How many pages does she have left to write?
**para**  Of the 84-page report Jennifer is required to submit, she has already finished writing a quarter. How many pages does she have left to write?  <sub>(sentence_join)</sub>
**formal**  GIVEN n=84, frac=Fraction(1, 4) — WHERE: Jennifer is required to submit a n-page report. She already finished writing frac of the report. — FIND: How many pages does she have left to write? → `answer = ?`
**break**  (n: 84→80) Jennifer is required to submit a 80-page report. She already finished writing a quarter of the report. How many pages does she have left to write?  → answer: **60** (was 63)

---

### 79. `gsms_0078` — answer: **75**

**base**  There are 6 periods in the day for a normal student but Jamal has to take 4 extra classes. Each class is 45 minutes long. He goes to class for 5 days a week. He then spends half of his weekly minutes each on Saturday and Sunday as extra learning time. How many hours a week does he spend learning?
**para**  A normal student has 6 periods in the day, but Jamal has to take 4 extra classes, each of them 45 minutes long, and he goes to class 5 days a week. On top of that he spends half of his weekly minutes on each of Saturday and Sunday as extra learning time. How many hours a week does he spend learning?  <sub>(sentence_join)</sub>
**formal**  GIVEN n1=6, n2=4, t=45, d=5, frac=Fraction(1, 2) — WHERE: There are n1 periods in the day for a normal student but Jamal has to take n2 extra classes. Each class is t minutes long. He goes to class for d days a week. He then spends frac of his weekly minutes each on Saturday and Sunday as extra learning time. — FIND: How many hours a week does he spend learning? → `answer = ?`
**break**  (n1: 6→8) There are 8 periods in the day for a normal student but Jamal has to take 4 extra classes. Each class is 45 minutes long. He goes to class for 5 days a week. He then spends half of his weekly minutes each on Saturday and Sunday as extra learning time. How many hours a week does he spend learning?  → answer: **90** (was 75)

---

### 80. `gsms_0079` — answer: **804**

**base**  John finished 1/5 of a 385 piece puzzle, and then started and finished another 727 piece puzzle within an hour. How many puzzle pieces did John place during that hour?
**para**  Within an hour, John finished 1/5 of a 385 piece puzzle and then started and finished another 727 piece puzzle. During that hour, how many puzzle pieces did John place?  <sub>(fronted_adverbial)</sub>
**formal**  GIVEN frac=Fraction(1, 5), n1=385, n2=727 — WHERE: John finished frac of a n1 piece puzzle, and then started and finished another n2 piece puzzle within an hour. — FIND: How many puzzle pieces did John place during that hour? → `answer = ?`
**break**  (n1: 385→380) John finished 1/5 of a 380 piece puzzle, and then started and finished another 727 piece puzzle within an hour. How many puzzle pieces did John place during that hour?  → answer: **803** (was 804)

---

### 81. `gsms_0080` — answer: **987**

**base**  The vending machines sell cookies for 75 cents and crackers for 100 cents. Mateo spent ¥1000 and got 4 bags of cookies and had 1% of his money left in change. How many crackers did he buy?
**para**  Cookies sell for 75 cents in the vending machines and crackers for 100 cents. Having spent ¥1000, Mateo came away with 4 bags of cookies and 1% of his money left in change. How many crackers did he buy?  <sub>(topicalization)</sub>
**formal**  GIVEN price1=75, price2=100, total=1000, n1=4, p=1 — WHERE: The vending machines sell cookies for price1 cents and crackers for price2 cents. Mateo spent ¥total and got n1 bags of cookies and had p% of his money left in change. — FIND: How many crackers did he buy? → `answer = ?`
**break**  (price1: 75→25) The vending machines sell cookies for 25 cents and crackers for 100 cents. Mateo spent ¥1000 and got 4 bags of cookies and had 1% of his money left in change. How many crackers did he buy?  → answer: **989** (was 987)

---

### 82. `gsms_0081` — answer: **48**

**base**  The Treasure hunt team hid 90 treats. The Johnson twins each found 12 treats. All the other treats except 18 were found by their friends. How many treats did the friends find?
**para**  90 treats were hidden by the Treasure hunt team. Each of the Johnson twins found 12 treats, and their friends found all the remaining treats except 18. How many treats did the friends find?  <sub>(passive_voice)</sub>
**formal**  GIVEN total=90, n=12, m=18 — WHERE: The Treasure hunt team hid total treats. The Johnson twins each found n treats. All the other treats except m were found by their friends. — FIND: How many treats did the friends find? → `answer = ?`
**break**  (total: 90→80) The Treasure hunt team hid 80 treats. The Johnson twins each found 12 treats. All the other treats except 18 were found by their friends. How many treats did the friends find?  → answer: **38** (was 48)

---

### 83. `gsms_0082` — answer: **79**

**base**  At the beginning of the party, there were 69 engineers and 66 teachers. After 45 minutes, 1/5 of the total number of people left. How many teachers are left if 29 engineers stayed at the party?
**para**  The party began with 69 engineers and 66 teachers present. 1/5 of the total number of people left after 45 minutes. If 29 engineers stayed at the party, how many teachers are left?  <sub>(existential_to_possessive)</sub>
**formal**  GIVEN m=69, w=66, frac=Fraction(1, 5), m_left=29 — WHERE: At the beginning of the party, there were m engineers and w teachers. After 45 minutes, frac of the total number of people left. — FIND: How many teachers are left if m_left engineers stayed at the party? → `answer = ?`
**break**  (m: 69→64) At the beginning of the party, there were 64 engineers and 66 teachers. After 45 minutes, 1/5 of the total number of people left. How many teachers are left if 29 engineers stayed at the party?  → answer: **75** (was 79)

---

### 84. `gsms_0083` — answer: **3208**

**base**  The Williams family is busy making biscuits. So far, they've made 7775 biscuits. They have 1774 blueberry biscuits, 2793 lemon biscuits, and some vanilla biscuits. How many vanilla biscuits have they made?
**para**  Busy making biscuits, the Williams family has produced 7775 of them so far: 1774 blueberry biscuits, 2793 lemon biscuits, and some vanilla ones. How many vanilla biscuits have they made?  <sub>(sentence_join)</sub>
**formal**  GIVEN total=7775, n1=1774, n2=2793 — WHERE: The Williams family is busy making biscuits. So far, they've made total biscuits. They have n1 blueberry biscuits, n2 lemon biscuits, and some vanilla biscuits. — FIND: How many vanilla biscuits have they made? → `answer = ?`
**break**  (total: 7775→7725) The Williams family is busy making biscuits. So far, they've made 7725 biscuits. They have 1774 blueberry biscuits, 2793 lemon biscuits, and some vanilla biscuits. How many vanilla biscuits have they made?  → answer: **3158** (was 3208)

---

### 85. `gsms_0084` — answer: **132**

**base**  Elise buys 8 shares of a stock for ¥22 each. The stock price increases 25% the first year Elise holds it, then decreases 40% in the second year. What is the final value of all Elise's shares?
**para**  Elise buys 8 shares of a stock at ¥22 each. In the first year she holds it, the stock price increases 25%. In the second year it decreases 40%. What is the final value of all Elise's shares?  <sub>(sentence_split)</sub>
**formal**  GIVEN n=8, p=22, r1=25, r2=40 — WHERE: Elise buys n shares of a stock for ¥p each. The stock price increases r1% the first year Elise holds it, then decreases r2% in the second year. — FIND: What is the final value of all Elise's shares? → `answer = ?`
**break**  (n: 8→6) Elise buys 6 shares of a stock for ¥22 each. The stock price increases 25% the first year Elise holds it, then decreases 40% in the second year. What is the final value of all Elise's shares?  → answer: **99** (was 132)

---

### 86. `gsms_0085` — answer: **15**

**base**  If Ahmed and Jennifer have 160 yards of fence between them, and they agree to split it with Jennifer getting 130 yards more than Ahmed, how much is left over for Ahmed?
**para**  The 160 yards of fence that Ahmed and Jennifer have between them is to be split so that Jennifer gets 130 yards more than Ahmed. How much is left over for Ahmed?  <sub>(passive_voice)</sub>
**formal**  GIVEN total=160, diff=130 — FIND: If Ahmed and Jennifer have total yards of fence between them, and they agree to split it with Jennifer getting diff yards more than Ahmed, how much is left over for Ahmed? → `answer = ?`
**break**  (total: 160→180) If Ahmed and Jennifer have 180 yards of fence between them, and they agree to split it with Jennifer getting 130 yards more than Ahmed, how much is left over for Ahmed?  → answer: **25** (was 15)

---

### 87. `gsms_0086` — answer: **79**

**base**  Nadia qualified for a spot on the swimming team, so she went shopping for some athletic gear. She bought a sports bra for €23, a pair of swimming sweatpants for €6 more than the sports bra cost, and a pair of athletic shoes that were originally €30 but were on sale for half price. She had a coupon for €2 off the package of €14 athletic headbands that she also bought. How much did she spend on athletic gear?
**para**  Having qualified for a spot on the swimming team, Nadia went shopping for some athletic gear. Into her basket went a sports bra at €23, a pair of swimming sweatpants costing €6 more than the sports bra, and a pair of athletic shoes marked down to half price from €30. She also bought a package of €14 athletic headbands, for which she had a coupon for €2 off. How much did she spend on athletic gear?  <sub>(topicalization)</sub>
**formal**  GIVEN p1=23, p2=6, p3=30, frac=Fraction(1, 2), d=2, p4=14 — WHERE: Nadia qualified for a spot on the swimming team, so she went shopping for some athletic gear. She bought a sports bra for €p1, a pair of swimming sweatpants for €p2 more than the sports bra cost, and a pair of athletic shoes that were originally €p3 but were on sale for frac price. She had a coupon for €d off the package of €p4 athletic headbands that she also bought. — FIND: How much did she spend on athletic gear? → `answer = ?`
**break**  (p1: 23→22) Nadia qualified for a spot on the swimming team, so she went shopping for some athletic gear. She bought a sports bra for €22, a pair of swimming sweatpants for €6 more than the sports bra cost, and a pair of athletic shoes that were originally €30 but were on sale for half price. She had a coupon for €2 off the package of €14 athletic headbands that she also bought. How much did she spend on athletic gear?  → answer: **77** (was 79)

---

### 88. `gsms_0087` — answer: **1**

**base**  The biggest waterslide at Splash World is 320 meters long, and people slide down at 80 meters/minute. The second biggest waterslide is 270 meters long, but steeper, so people slide down at 90 meters/minute. How much longer does it take to ride the biggest slide compared to the second biggest slide?
**para**  At Splash World, the biggest waterslide is 320 meters long and people slide down it at 80 meters/minute, while the second biggest is 270 meters long but steeper, with people sliding down at 90 meters/minute. Compared with the second biggest slide, how much longer does it take to ride the biggest one?  <sub>(sentence_join)</sub>
**formal**  GIVEN l1=320, s1=80, l2=270, s2=90 — WHERE: The biggest waterslide at Splash World is l1 meters long, and people slide down at s1 meters/minute. The second biggest waterslide is l2 meters long, but steeper, so people slide down at s2 meters/minute. — FIND: How much longer does it take to ride the biggest slide compared to the second biggest slide? → `answer = ?`
**break**  (s1: 80→40) The biggest waterslide at Splash World is 320 meters long, and people slide down at 40 meters/minute. The second biggest waterslide is 270 meters long, but steeper, so people slide down at 90 meters/minute. How much longer does it take to ride the biggest slide compared to the second biggest slide?  → answer: **5** (was 1)

---

### 89. `gsms_0088` — answer: **40**

**base**  Fatima can run three times faster than she can walk, but she can skip at a rate of speed that is half as fast as she can run. If she can skip at 3 miles per hour, how many miles can she travel in 10 hours if she spends half of the time running and half of the time walking?
**para**  Fatima's running speed is three times her walking speed, and her skipping speed is half as fast as her running speed. She can skip at 3 miles per hour. If she spends half of her time running and half of it walking, how many miles can she travel in 10 hours?  <sub>(nominalization)</sub>
**formal**  GIVEN mult_run=3, frac_skip=Fraction(1, 2), skip_speed=3, total_time=10, frac_run=Fraction(1, 2), frac_walk=Fraction(1, 2) — WHERE: Fatima can run mult_run faster than she can walk, but she can skip at a rate of speed that is frac_skip as fast as she can run. — FIND: If she can skip at skip_speed miles per hour, how many miles can she travel in total_time hours if she spends frac_run of the time running and frac_walk of the time walking? → `answer = ?`
**break**  (skip_speed: 3→6) Fatima can run three times faster than she can walk, but she can skip at a rate of speed that is half as fast as she can run. If she can skip at 6 miles per hour, how many miles can she travel in 10 hours if she spends half of the time running and half of the time walking?  → answer: **80** (was 40)

---

### 90. `gsms_0089` — answer: **161**

**base**  Hiroshi hires a limousine from 4 PM to 8 PM. He gets 1 hours free. The first paid hour is $23 and each hour after that is triple the cost. How much did he pay?
**para**  Hiroshi hires a limousine for the stretch from 4 PM to 8 PM, of which 1 hours are free. The first paid hour is $23 and each hour after that is triple the cost. How much did he pay?  <sub>(relative_clause)</sub>
**formal**  GIVEN start_time=4, end_time=8, free_hours=1, first_hour_cost=23, multiplier=3 — WHERE: Hiroshi hires a limousine from start_time PM to end_time PM. He gets free_hours hours free. The first paid hour is $first_hour_cost and each hour after that is multiplier the cost. — FIND: How much did he pay? → `answer = ?`
**break**  (start_time: 4→3) Hiroshi hires a limousine from 3 PM to 8 PM. He gets 1 hours free. The first paid hour is $23 and each hour after that is triple the cost. How much did he pay?  → answer: **230** (was 161)

---

### 91. `gsms_0090` — answer: **36**

**base**  Tomas has four times as many Snapchat friends as Ali. John has triple as many friends on Snapchat as Ali. If Tomas has 48 friends on Snapchat, how many Snapchat friends does John have?
**para**  The number of Snapchat friends Tomas has is four times the number Ali has, and John has triple as many friends on Snapchat as Ali. Given that Tomas has 48 friends on Snapchat, how many Snapchat friends does John have?  <sub>(nominalization)</sub>
**formal**  GIVEN mult1=4, mult2=3, n=48 — WHERE: Tomas has mult1 as many Snapchat friends as Ali. John has mult2 as many friends on Snapchat as Ali. — FIND: If Tomas has n friends on Snapchat, how many Snapchat friends does John have? → `answer = ?`
**break**  (n: 48→44) Tomas has four times as many Snapchat friends as Ali. John has triple as many friends on Snapchat as Ali. If Tomas has 44 friends on Snapchat, how many Snapchat friends does John have?  → answer: **33** (was 36)

---

### 92. `gsms_0091` — answer: **548**

**base**  At Omar's house, there are three times as many kiwis as bananas. He has a total of 93 bananas in his house. Omar bought 96 more bananas at the store and 16 fewer kiwis than the number of bananas. Find the combined total of the number of kiwis and bananas Omar has in the house?
**para**  Omar's house holds three times as many kiwis as bananas, and there are 93 bananas in it in total. At the store he bought 96 more bananas, and 16 fewer kiwis than the number of bananas. What is the combined total of the number of kiwis and bananas Omar has in the house?  <sub>(imperative_to_wh)</sub>
**formal**  GIVEN mult=3, n=93, m=96, k=16 — WHERE: At Omar's house, there are mult as many kiwis as bananas. He has a total of n bananas in his house. Omar bought m more bananas at the store and k fewer kiwis than the number of bananas. — FIND: Find the combined total of the number of kiwis and bananas Omar has in the house? → `answer = ?`
**break**  (n: 93→92) At Omar's house, there are three times as many kiwis as bananas. He has a total of 92 bananas in his house. Omar bought 96 more bananas at the store and 16 fewer kiwis than the number of bananas. Find the combined total of the number of kiwis and bananas Omar has in the house?  → answer: **544** (was 548)

---

### 93. `gsms_0092` — answer: **75**

**base**  David slew 300 giants with his mighty war hammer, while John, using a spear, slew 1/3 as many giants as David. Using a bone dagger, Robert slew three times as many giants as John. But Wei, having forgotten his axe at home, slew a quarter as many giants as Robert using a wooden spoon. How many Giants has Wei slain?
**para**  With his mighty war hammer, David slew 300 giants. John slew 1/3 as many giants as David, using a spear, and Robert slew three times as many giants as John with a bone dagger. Wei, who had forgotten his axe at home and had to make do with a wooden spoon, slew a quarter as many giants as Robert. How many giants has Wei slain?  <sub>(clause_reorder)</sub>
**formal**  GIVEN n1=300, frac1=Fraction(1, 3), mult1=3, frac2=Fraction(1, 4) — WHERE: David slew n1 giants with his mighty war hammer, while John, using a spear, slew frac1 as many giants as David. Using a bone dagger, Robert slew mult1 as many giants as John. But Wei, having forgotten his axe at home, slew frac2 as many giants as Robert using a wooden spoon. — FIND: How many Giants has Wei slain? → `answer = ?`
**break**  (frac1: 1/3→1/5) David slew 300 giants with his mighty war hammer, while John, using a spear, slew 1/5 as many giants as David. Using a bone dagger, Robert slew three times as many giants as John. But Wei, having forgotten his axe at home, slew a quarter as many giants as Robert using a wooden spoon. How many Giants has Wei slain?  → answer: **45** (was 75)

---

### 94. `gsms_0093` — answer: **126**

**base**  When the water is cold Diego swims a mile in 16 minutes. When the water is warm Diego swims a mile in 2 minutes more than double as long. How much longer does Diego take to swim 7 miles on a hot day than a cold day?
**para**  Diego swims a mile in 16 minutes when the water is cold. In warm water he needs 2 minutes more than double as long. Over 7 miles, how much longer does a hot day take him than a cold day?  <sub>(clause_reorder)</sub>
**formal**  GIVEN t1=16, n=2, mult=2, d=7 — WHERE: When the water is cold Diego swims a mile in t1 minutes. When the water is warm Diego swims a mile in n minutes more than mult as long. — FIND: How much longer does Diego take to swim d miles on a hot day than a cold day? → `answer = ?`
**break**  (t1: 16→15) When the water is cold Diego swims a mile in 15 minutes. When the water is warm Diego swims a mile in 2 minutes more than double as long. How much longer does Diego take to swim 7 miles on a hot day than a cold day?  → answer: **119** (was 126)

---

### 95. `gsms_0094` — answer: **20**

**base**  800 people apply for a job at Facebook. Of the people that apply, only 30% receive interviews. Of those who receive interviews, 25% receive a job offer. Of those who receive a job offer, a third of the people accept the position. How many people accept the position?
**para**  A job at Facebook draws 800 applicants. Interviews go to only 30% of those who apply, job offers to 25% of those who are interviewed, and of the people who receive a job offer, a third accept the position. How many people accept the position?  <sub>(sentence_join)</sub>
**formal**  GIVEN n=800, p1=30, p2=25, frac=Fraction(1, 3) — WHERE: n people apply for a job at Facebook. Of the people that apply, only p1% receive interviews. Of those who receive interviews, p2% receive a job offer. Of those who receive a job offer, frac of the people accept the position. — FIND: How many people accept the position? → `answer = ?`
**break**  (n: 800→840) 840 people apply for a job at Facebook. Of the people that apply, only 30% receive interviews. Of those who receive interviews, 25% receive a job offer. Of those who receive a job offer, a third of the people accept the position. How many people accept the position?  → answer: **21** (was 20)

---

### 96. `gsms_0095` — answer: **87**

**base**  There are 14 students playing table tennis and triple that number playing volleyball. There are 19 boys and 12 girls playing football. If each student only participates in one group, how many students are there in total?
**para**  14 students play table tennis and triple that number play volleyball, while 19 boys and 12 girls play football. If each student only participates in one group, how many students are there in total?  <sub>(existential_to_possessive)</sub>
**formal**  GIVEN n1=14, mult=3, n2=19, n3=12 — WHERE: There are n1 students playing table tennis and mult that number playing volleyball. There are n2 boys and n3 girls playing football. — FIND: If each student only participates in one group, how many students are there in total? → `answer = ?`
**break**  (n1: 14→13) There are 13 students playing table tennis and triple that number playing volleyball. There are 19 boys and 12 girls playing football. If each student only participates in one group, how many students are there in total?  → answer: **83** (was 87)

---

### 97. `gsms_0096` — answer: **5**

**base**  Nadia's dad said that she had £300 budgeted for her birthday party. She wants to make sure she and her friends all get to play one round of bowling, have £7 in pinball tokens, and get to ride the go-karts four times. A round of bowling is £7. The go-karts cost £9 a ride. How many friends can she invite?
**para**  £300 has been budgeted for Nadia's birthday party, her dad said. She wants to make sure that she and her friends all get one round of bowling, £7 in pinball tokens, and four times on the go-karts. A round of bowling costs £7, and the go-karts cost £9 a ride. How many friends can she invite?  <sub>(passive_voice)</sub>
**formal**  GIVEN budget=300, tokens=7, times=4, cost1=7, cost2=9 — WHERE: Nadia's dad said that she had £budget budgeted for her birthday party. She wants to make sure she and her friends all get to play one round of bowling, have £tokens in pinball tokens, and get to ride the go-karts times. A round of bowling is £cost1. The go-karts cost £cost2 a ride. — FIND: How many friends can she invite? → `answer = ?`
**break**  (budget: 300→200) Nadia's dad said that she had £200 budgeted for her birthday party. She wants to make sure she and her friends all get to play one round of bowling, have £7 in pinball tokens, and get to ride the go-karts four times. A round of bowling is £7. The go-karts cost £9 a ride. How many friends can she invite?  → answer: **3** (was 5)

---

### 98. `gsms_0097` — answer: **2142**

**base**  Linda operates the cash register exactly five times as fast as her less-experienced colleague Ines. Daily, Ines processes 51 customers. What is the total weekly production for the two if they work all days of the week?
**para**  Operating the cash register, Linda is exactly five times as fast as her less-experienced colleague Ines, who processes 51 customers daily. If they work all days of the week, what is the total weekly production for the two?  <sub>(relative_clause)</sub>
**formal**  GIVEN mult=5, n=51 — WHERE: Linda operates the cash register exactly mult as fast as her less-experienced colleague Ines. Daily, Ines processes n customers. — FIND: What is the total weekly production for the two if they work all days of the week? → `answer = ?`
**break**  (n: 51→50) Linda operates the cash register exactly five times as fast as her less-experienced colleague Ines. Daily, Ines processes 50 customers. What is the total weekly production for the two if they work all days of the week?  → answer: **2100** (was 2142)

---

### 99. `gsms_0098` — answer: **10000**

**base**  A new virus infects 16 people. Every day, each infected person infects 4 others. How many people are infected after 4 days?
**para**  16 people are infected by a new virus. Each infected person infects 4 others every day. After 4 days, how many people are infected?  <sub>(passive_voice)</sub>
**formal**  GIVEN n0=16, r=4, d=4 — WHERE: A new virus infects n0 people. Every day, each infected person infects r others. — FIND: How many people are infected after d days? → `answer = ?`
**break**  (n0: 16→15) A new virus infects 15 people. Every day, each infected person infects 4 others. How many people are infected after 4 days?  → answer: **9375** (was 10000)

---

### 100. `gsms_0099` — answer: **59**

**base**  If Marcus eats 5 eggs a day for 102 days and then increases it to 6 eggs a day for 33 days, how many dozens of eggs will Marcus need for 135 days?
**para**  For 102 days Marcus eats 5 eggs a day, and for the 33 days after that he steps it up to 6 eggs a day. Over the whole 135 days, how many dozens of eggs will Marcus need?  <sub>(fronted_adverbial)</sub>
**formal**  GIVEN n1=5, d1=102, n2=6, d2=33, d3=135 — FIND: If Marcus eats n1 eggs a day for d1 days and then increases it to n2 eggs a day for d2 days, how many dozens of eggs will Marcus need for d3 days? → `answer = ?`
**break**  (n1: 5→3) If Marcus eats 3 eggs a day for 102 days and then increases it to 6 eggs a day for 33 days, how many dozens of eggs will Marcus need for 135 days?  → answer: **42** (was 59)

---
