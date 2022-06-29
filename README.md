# stir_slurry_stir
project die LMAO but don't throw letter

Problem Statement:
- We have 3 tanks, 200kg of slurry capacity each
- When there is only 60kg of slurry in a tank, we can "switch tanks" by shifting the remaining 60kg to the next tank
- An empty tank needs to be cleaned (1hr), before it can then be filled (with 2 drums) (1.5hr). The 2 drums can only fill one tank at a time.
- Assume some incoming WIP (wafers) and some consumption amount for a few recipes over the next few hours. Our goal is to decide whether to pre-emptively "switch tanks" so that the incoming WIP doesn't get stuck without slurry.


States (in no ordering for tanks, can sort): 
- Tank down for cleaning (num periods left)
- Tank down for filing (num periods left)
- Tank current slurry holding
- Accumulated penalty
