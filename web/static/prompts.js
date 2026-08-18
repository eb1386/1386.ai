/* Suggestion pool for the empty state.
   Built combinatorially rather than hand-listed, and deliberately limited to
   what a 521M model answers decently: definitions, plain explanations and
   short writing. No dates, counts, arithmetic, code or niche trivia, because
   those are exactly where it invents things.

   Plurality is tagged, never inferred: "glass" ends in s and is singular,
   "savings" does not and is plural. Patterns agree off the tag. */

// singular and mass nouns, article included so patterns just interpolate
const ONE = [
  "gravity", "photosynthesis", "the water cycle", "evaporation", "electricity",
  "magnetism", "friction", "sound", "an echo", "a rainbow", "thunder",
  "lightning", "wind", "rain", "snow", "fog", "soil", "evolution", "DNA",
  "the immune system", "the heart", "the brain", "blood", "sleep", "memory",
  "stress", "exercise", "nutrition", "protein", "sugar", "caffeine", "oxygen",
  "the atmosphere", "the ozone layer", "climate change", "recycling",
  "pollution", "renewable energy", "solar power", "wind power", "concrete",
  "steel", "glass", "plastic", "paper", "wood", "cotton", "wool", "leather",
  "rubber", "the internet", "wifi", "email", "encryption", "software",
  "hardware", "memory in a computer", "a hard drive", "a processor",
  "a keyboard", "a printer", "a camera", "a microphone", "a speaker",
  "a television", "a radio", "a telephone", "a smartphone", "an app",
  "a website", "a search engine", "social media", "music", "rhythm",
  "melody", "harmony", "a guitar", "a piano", "a drum", "an orchestra",
  "opera", "ballet", "theatre", "cinema", "photography", "painting",
  "sculpture", "architecture", "poetry", "advertising", "money", "teamwork",
  "leadership", "communication", "negotiation", "football", "basketball",
  "tennis", "swimming", "running", "cycling", "chess", "gardening",
  "cooking", "baking", "bread", "cheese", "coffee", "tea", "chocolate",
  "honey", "salt", "fruit", "rice", "pasta", "soup", "breakfast",
  "the moon", "the sun", "the ocean", "kindness", "honesty", "patience",
  "courage", "curiosity", "a door hinge", "a zipper", "a button", "a lock",
  "a key", "a lightbulb", "a refrigerator", "a microwave", "an oven",
  "a washing machine", "a vacuum cleaner", "an umbrella", "a mirror",
  "a magnet", "a compass", "a map", "a thermometer", "a ladder", "a hammer",
  "a screwdriver", "a saw", "a screw", "glue", "tape", "a pencil", "a pen",
  "ink", "a notebook", "a backpack", "a traffic light", "a roundabout",
  "a lighthouse", "a windmill", "a greenhouse", "a beehive", "an anthill",
  "a bird's nest", "a spider web", "a seed", "a root", "a leaf", "bark",
  "moss", "a mushroom", "pollen", "nectar", "a wheel", "a bridge",
  "a tunnel", "a skyscraper", "a computer", "a password", "a novel",
  "a library", "a museum", "a newspaper", "a bank", "a contract", "a law",
  "a court", "a jury", "a puzzle", "a restaurant", "a market", "a train",
  "a bus", "a car", "a bicycle", "a motorcycle", "a boat", "a ship",
  "a submarine", "an airplane", "a helicopter", "a rocket", "a satellite",
  "a star", "a planet", "a comet", "an asteroid", "a galaxy", "a telescope",
  "an astronaut", "a space station", "a time zone", "a calendar", "a clock",
  "a holiday", "a wedding", "a family", "a career", "an interview",
  "a budget", "a habit", "a dream", "a vitamin", "a battery", "an engine",
  "a gear", "a lever", "a pulley", "a cell", "a virus", "a vaccine",
  "a fossil", "a glacier", "a desert", "a rainforest", "a coral reef",
  "a river", "a mountain", "a rock", "a mineral", "a volcano",
  "an earthquake", "a cloud", "a video game",
];

// plurals: patterns switch to are/do off this list
const MANY = [
  "the tides", "dinosaurs", "bacteria", "antibiotics", "the lungs",
  "muscles", "bones", "savings", "taxes", "the police", "firefighters",
  "doctors", "nurses", "teachers", "farmers", "chefs", "engineers",
  "scientists", "artists", "athletes", "coaches", "referees", "spices",
  "herbs", "vegetables", "headphones", "scissors", "shoes", "the seasons",
  "families", "libraries", "museums", "newspapers", "clouds", "rivers",
  "mountains", "stars", "planets", "galaxies", "trains", "cars",
];

const CONCEPT_PATTERNS = [
  (t, pl) => `What ${pl ? "are" : "is"} ${t}?`,
  (t) => `Explain ${t} in simple terms.`,
  (t, pl) => `How ${pl ? "do" : "does"} ${t} work?`,
  (t, pl) => `Why ${pl ? "do" : "does"} ${t} matter?`,
  (t) => `Describe ${t} in a few sentences.`,
  (t) => `Write a short paragraph about ${t}.`,
  (t) => `What should I know about ${t}?`,
  (t) => `Give me a simple explanation of ${t}.`,
  (t) => `Tell me something interesting about ${t}.`,
  (t) => `How would you explain ${t} to a beginner?`,
  (t, pl) => `What ${pl ? "are" : "is"} ${t} used for?`,
  (t) => `Why do people care about ${t}?`,
  (t) => `Sum up ${t} in a few sentences.`,
  (t) => `What makes ${t} interesting?`,
  (t) => `Give me a short overview of ${t}.`,
  (t, pl) => `In plain language, what ${pl ? "are" : "is"} ${t}?`,
];

// full clauses, so agreement is already baked in
const PHENOMENA = [
  "the ocean is salty", "the sky is blue", "leaves change colour",
  "ice floats on water", "bread rises", "onions make you cry",
  "metal feels colder than wood", "the moon changes shape",
  "we see lightning before we hear thunder", "soap cleans things",
  "hot air rises", "birds migrate", "cats purr", "dogs pant",
  "we yawn", "we sneeze", "we get goosebumps", "we dream",
  "we need sleep", "exercise makes you tired", "spicy food burns",
  "sugar tastes sweet", "coffee wakes you up", "hair turns grey",
  "cuts heal", "we get hungry", "we get thirsty", "flowers smell nice",
  "bees make honey", "trees lose their leaves", "grass is green",
  "the wind blows", "it rains", "rainbows appear", "snow is white",
  "deserts are dry", "mountains are cold at the top",
  "rivers flow to the sea", "the tide comes in", "shells wash up on beaches",
  "sand is soft", "mirrors reflect", "glass is transparent",
  "magnets stick to metal", "balloons float", "boats float",
  "planes stay in the air", "wheels make things easier to move",
  "bridges do not fall down", "buildings need foundations", "paint dries",
  "wood floats", "iron rusts", "milk goes sour", "food goes bad",
  "fridges keep food fresh", "ovens cook food", "candles melt",
  "fire needs air", "smoke rises", "ice melts", "water boils",
  "steam is hot", "the sun sets", "days get shorter in winter",
  "the seasons change", "some animals hibernate",
  "fish can breathe underwater", "birds can fly", "spiders spin webs",
  "ants live in colonies", "plants need sunlight", "seeds sprout",
  "roots grow downwards", "lemons are sour", "chillies are hot",
  "salt preserves food", "yeast makes dough rise", "cheese has holes",
  "popcorn pops", "eggs harden when cooked", "toast turns brown",
  "tea leaves colour the water", "oil and water do not mix",
  "sound travels", "echoes happen", "music can change your mood",
  "we remember songs easily", "we forget things",
];

const PHENOMENA_PATTERNS = [
  (t) => `Why ${t}?`,
  (t) => `Explain why ${t}.`,
  (t) => `Can you explain simply why ${t}?`,
  (t) => `In a short paragraph, explain why ${t}.`,
];

const STORY_SEEDS = [
  "a lighthouse keeper", "a night baker", "a village blacksmith",
  "a lonely astronaut", "a stray cat", "an old fisherman",
  "a girl who collects stones", "a boy who is afraid of the dark",
  "a retired postman", "a travelling musician", "a mountain guide",
  "a librarian who cannot sleep", "a gardener in winter",
  "a train conductor", "a bus driver on a quiet route",
  "a shopkeeper who never closes", "a clockmaker", "a beekeeper",
  "a shepherd and a storm", "a farmer waiting for rain",
  "a child and a stray dog", "two friends who lose a map",
  "a family moving house", "a sailor coming home",
  "a painter who runs out of paint", "a chef with one ingredient left",
  "a runner in the last mile", "a swimmer crossing a lake",
  "a climber who turns back", "a diver who finds something",
  "an inventor whose machine fails", "a scientist who is wrong",
  "a teacher on the last day of term", "a student who oversleeps",
  "a nurse on a night shift", "a firefighter and a cat in a tree",
  "a mechanic and a car that will not start", "a pilot in fog",
  "a photographer chasing light", "a writer with no ideas",
  "a bookshop at closing time", "a market before dawn",
  "a house by the sea", "a cabin in the snow", "a garden at night",
  "a bridge in the rain", "an empty theatre", "a lost umbrella",
  "a letter that arrives late", "a key with no lock",
  "a lighthouse without a keeper", "a clock that runs backwards",
  "a door that was never opened", "a road with no signs",
  "a boat with no name", "a train that never stops",
  "a town where it always rains", "an island with one tree",
  "a forest after a fire", "a river that dries up",
  "the last day of summer", "the first day of snow",
  "a very long walk home", "a small act of kindness",
  "a promise that is kept", "a secret that is shared",
  "an old photograph", "a song nobody remembers",
  "a map drawn from memory", "a stranger who helps",
];

const STORY_PATTERNS = [
  (t) => `Write a short story about ${t}.`,
  (t) => `Write a few paragraphs about ${t}.`,
  (t) => `Tell me a short story about ${t}.`,
  (t) => `Write the opening of a story about ${t}.`,
];

const HOWTOS = [
  "make a cup of tea", "boil an egg", "bake bread", "cook rice",
  "make soup", "make a sandwich", "keep vegetables fresh",
  "plan meals for the week", "write a shopping list", "save money",
  "make a budget", "prepare for an interview", "write a cover letter",
  "introduce yourself", "give a short speech", "listen better",
  "apologise properly", "make a new friend", "keep in touch with people",
  "be more patient", "handle stress", "sleep better", "wake up earlier",
  "build a habit", "stop procrastinating", "focus while studying",
  "take good notes", "revise for a test", "read more books",
  "learn a new skill", "practise an instrument", "start running",
  "stretch properly", "stay motivated", "set a goal", "organise a desk",
  "tidy a room", "pack a suitcase", "plan a trip", "read a map",
  "use a compass", "care for a houseplant", "start a garden",
  "water plants correctly", "look after a pet", "train a dog",
  "wash clothes properly", "remove a stain", "sew a button",
  "fix a squeaky door", "change a lightbulb", "unclog a sink",
  "hang a picture straight", "sharpen a knife", "clean a window",
  "fold clothes neatly", "wrap a present", "write a thank-you note",
  "tell a good story", "take a better photograph", "draw from life",
  "write a poem",
];

const HOWTO_PATTERNS = [
  (t) => `How do I ${t}?`,
  (t) => `What is the best way to ${t}?`,
  (t) => `Give me some tips on how to ${t}.`,
  (t) => `Explain step by step how to ${t}.`,
];

const WRITING = [
  "the changing seasons", "a rainy afternoon", "your favourite meal",
  "a place you feel calm", "the sound of the sea", "an early morning",
  "a busy street", "a quiet library", "an old house",
  "the smell of fresh bread", "a long journey", "coming home",
  "learning something difficult", "making a mistake", "asking for help",
  "being patient", "being brave", "keeping a promise", "saying goodbye",
  "starting again", "the value of friendship", "why kindness matters",
  "why reading matters", "why exercise matters", "why sleep matters",
  "why curiosity matters", "the importance of teamwork",
  "the importance of honesty", "why we tell stories", "why music matters",
  "life in a small town", "life in a big city", "growing up",
  "spending time outdoors", "cooking for other people",
  "the first day of school", "a family tradition", "a favourite season",
  "a walk in the woods", "watching a storm", "looking at the stars",
  "the ocean at night", "a garden in spring", "an autumn morning",
  "a winter evening", "a summer afternoon", "an unexpected gift",
  "a small victory", "a lesson learned", "a change of plan",
];

const WRITING_PATTERNS = [
  (t) => `Write a short essay about ${t}.`,
  (t) => `Write a paragraph about ${t}.`,
  (t) => `Write a few sentences about ${t}.`,
  (t) => `Describe ${t}.`,
];

const tag = (list, plural) => list.map((name) => ({ name, plural }));

/* Groups are addressed by a flat index so three can be drawn without ever
   building the whole list in memory. */
const GROUPS = [
  { subjects: [...tag(ONE, false), ...tag(MANY, true)], patterns: CONCEPT_PATTERNS },
  { subjects: tag(PHENOMENA, false), patterns: PHENOMENA_PATTERNS },
  { subjects: tag(STORY_SEEDS, false), patterns: STORY_PATTERNS },
  { subjects: tag(HOWTOS, false), patterns: HOWTO_PATTERNS },
  { subjects: tag(WRITING, false), patterns: WRITING_PATTERNS },
];

const PROMPT_TOTAL = GROUPS.reduce(
  (n, g) => n + g.subjects.length * g.patterns.length, 0);

function promptAt(index) {
  let n = ((index % PROMPT_TOTAL) + PROMPT_TOTAL) % PROMPT_TOTAL;
  for (const g of GROUPS) {
    const size = g.subjects.length * g.patterns.length;
    if (n < size) {
      const s = g.subjects[Math.floor(n / g.patterns.length)];
      return { text: g.patterns[n % g.patterns.length](s.name, s.plural), subject: s.name };
    }
    n -= size;
  }
  return { text: "What is gravity?", subject: "gravity" };
}

// three at a time, never repeating a subject within the set
function pickPrompts(count) {
  const out = [], seen = new Set();
  let guard = 0;
  while (out.length < count && guard++ < count * 40) {
    const p = promptAt(Math.floor(Math.random() * PROMPT_TOTAL));
    if (seen.has(p.subject)) continue;
    seen.add(p.subject);
    out.push(p.text);
  }
  return out;
}

window.PROMPTS = { total: PROMPT_TOTAL, pick: pickPrompts, at: promptAt };
