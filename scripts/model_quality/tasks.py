"""The task set: 12 functions to write, 8 bugs to fix, 12 word problems, 6 JSON extractions, 6 tool calls. Written for
this comparison (not a public benchmark). Each task carries how it is scored; `reference` answers check the scorers."""

CODE = [  # (function, spec, tests (asserts), reference solution)
    ("merge_intervals", "merge_intervals(intervals) takes a list of [start, end] pairs (unsorted, inclusive) and returns "
     "the merged, non-overlapping intervals sorted by start. Touching intervals ([1,4] and [4,5]) merge.",
     ["merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]", "merge_intervals([[1,4],[4,5]]) == [[1,5]]",
      "merge_intervals([]) == []", "merge_intervals([[5,7],[1,2]]) == [[1,2],[5,7]]"],
     "def merge_intervals(intervals):\n    out = []\n    for s, e in sorted(intervals):\n        if out and s <= out[-1][1]:\n"
     "            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out"),
    ("is_balanced", "is_balanced(s) returns True when the brackets (), [] and {} in the string s are balanced and "
     "properly nested; other characters are ignored.",
     ["is_balanced('([]{})') is True", "is_balanced('([)]') is False", "is_balanced('a(b)c') is True",
      "is_balanced('') is True", "is_balanced('((') is False", "is_balanced('}') is False"],
     "def is_balanced(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    st = []\n    for c in s:\n        if c in '([{':\n"
     "            st.append(c)\n        elif c in pairs:\n            if not st or st.pop() != pairs[c]:\n                return False\n"
     "    return not st"),
    ("roman_to_int", "roman_to_int(s) converts a Roman numeral string (I, V, X, L, C, D, M, with subtractive forms like "
     "IV and CM) to an integer.",
     ["roman_to_int('III') == 3", "roman_to_int('IV') == 4", "roman_to_int('IX') == 9", "roman_to_int('LVIII') == 58",
      "roman_to_int('MCMXCIV') == 1994"],
     "def roman_to_int(s):\n    v = dict(I=1, V=5, X=10, L=50, C=100, D=500, M=1000)\n    t = 0\n    for i, c in enumerate(s):\n"
     "        t += -v[c] if i + 1 < len(s) and v[c] < v[s[i + 1]] else v[c]\n    return t"),
    ("top_k_words", "top_k_words(words, k) returns the k most frequent strings in the list words, most frequent first; "
     "ties are broken alphabetically.",
     ["top_k_words(['b','a','b','c','a','b'], 2) == ['b','a']", "top_k_words(['x','y','z'], 2) == ['x','y']",
      "top_k_words(['apple','bob','apple','bob','cat'], 3) == ['apple','bob','cat']"],
     "from collections import Counter\ndef top_k_words(words, k):\n    c = Counter(words)\n"
     "    return sorted(c, key=lambda w: (-c[w], w))[:k]"),
    ("parse_duration", "parse_duration(s) converts a duration string such as '1h30m', '45s', '2h5s' or '10m' (optional "
     "hours, minutes and seconds parts, in that order, each digits followed by h, m or s) to a number of seconds; the "
     "empty string is 0.",
     ["parse_duration('1h30m') == 5400", "parse_duration('45s') == 45", "parse_duration('2h5s') == 7205",
      "parse_duration('10m') == 600", "parse_duration('') == 0"],
     "import re\ndef parse_duration(s):\n    return sum(int(n) * {'h': 3600, 'm': 60, 's': 1}[u] for n, u in re.findall(r'(\\d+)([hms])', s))"),
    ("flatten", "flatten(nested) returns a flat list of the non-list items of an arbitrarily nested list, in order.",
     ["flatten([1,[2,[3,[4]],5]]) == [1,2,3,4,5]", "flatten([]) == []", "flatten([[],[[]]]) == []",
      "flatten(['a',['b']]) == ['a','b']"],
     "def flatten(nested):\n    out = []\n    for x in nested:\n        out.extend(flatten(x) if isinstance(x, list) else [x])\n    return out"),
    ("longest_common_prefix", "longest_common_prefix(strs) returns the longest string that is a prefix of every string "
     "in the list strs ('' for an empty list).",
     ["longest_common_prefix(['flower','flow','flight']) == 'fl'", "longest_common_prefix(['dog','racecar','car']) == ''",
      "longest_common_prefix([]) == ''", "longest_common_prefix(['a']) == 'a'"],
     "import os\ndef longest_common_prefix(strs):\n    return os.path.commonprefix(strs) if strs else ''"),
    ("rotate_clockwise", "rotate_clockwise(matrix) returns the matrix (a list of equal-length rows, possibly not square) "
     "rotated 90 degrees clockwise, as a new list of lists.",
     ["rotate_clockwise([[1,2],[3,4]]) == [[3,1],[4,2]]", "rotate_clockwise([[1,2,3]]) == [[1],[2],[3]]",
      "rotate_clockwise([]) == []"],
     "def rotate_clockwise(matrix):\n    return [list(r) for r in zip(*matrix[::-1])]"),
    ("wrap_text", "wrap_text(text, width) greedily wraps the words of text (separated by single spaces) into lines of "
     "at most width characters, words joined by single spaces; a word longer than width gets a line of its own. "
     "Returns the list of lines ([] for empty text).",
     ["wrap_text('the quick brown fox', 10) == ['the quick','brown fox']", "wrap_text('hello', 3) == ['hello']",
      "wrap_text('a b c', 1) == ['a','b','c']", "wrap_text('', 5) == []"],
     "def wrap_text(text, width):\n    lines = []\n    for w in text.split():\n        if lines and len(lines[-1]) + 1 + len(w) <= width:\n"
     "            lines[-1] += ' ' + w\n        else:\n            lines.append(w)\n    return lines"),
    ("dedupe", "dedupe(items) returns the items with repeats removed, keeping each first occurrence, in order.",
     ["dedupe([3,1,3,2,1]) == [3,1,2]", "dedupe([]) == []", "dedupe(['a','A','a']) == ['a','A']"],
     "def dedupe(items):\n    return list(dict.fromkeys(items))"),
    ("left_insert_index", "left_insert_index(a, x) returns the leftmost index at which x could be inserted into the "
     "sorted list a keeping it sorted (as bisect.bisect_left does), using binary search.",
     ["left_insert_index([1,2,2,3], 2) == 1", "left_insert_index([], 5) == 0", "left_insert_index([1,3], 4) == 2",
      "left_insert_index([1,3], 0) == 0"],
     "def left_insert_index(a, x):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        m = (lo + hi) // 2\n"
     "        if a[m] < x:\n            lo = m + 1\n        else:\n            hi = m\n    return lo"),
    ("LRUCache", "class LRUCache(capacity) with get(key), which returns the value or -1 when absent, and put(key, value); "
     "when full, put evicts the least recently used key (get and put both count as use).",
     ["[r for i, r in enumerate((lambda c: [c.put(1,1), c.put(2,2), c.get(1), c.put(3,3), c.get(2), c.put(4,4), "
      "c.get(1), c.get(3), c.get(4)])(LRUCache(2))) if i in (2, 4, 6, 7, 8)] == [1, -1, -1, 3, 4]"],
     "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, capacity):\n        self.c, self.d = capacity, OrderedDict()\n"
     "    def get(self, k):\n        if k not in self.d:\n            return -1\n        self.d.move_to_end(k)\n        return self.d[k]\n"
     "    def put(self, k, v):\n        self.d[k] = v\n        self.d.move_to_end(k)\n        if len(self.d) > self.c:\n"
     "            self.d.popitem(last=False)"),
]

BUGS = [  # (function, buggy code, symptom, tests, reference fix)
    ("sum_range", "def sum_range(a, b):\n    \"\"\"Sum of the integers from a to b inclusive.\"\"\"\n    total = 0\n"
     "    for i in range(a, b):\n        total += i\n    return total", "sum_range(1, 5) returns 10 but should return 15.",
     ["sum_range(1, 5) == 15", "sum_range(3, 3) == 3", "sum_range(-2, 2) == 0"],
     "def sum_range(a, b):\n    return sum(range(a, b + 1))"),
    ("add_tag", "def add_tag(tag, tags=[]):\n    tags.append(tag)\n    return tags",
     "Calling add_tag('a') and then add_tag('b') returns ['a', 'b'] the second time instead of ['b'].",
     ["add_tag('a') == ['a']", "add_tag('b') == ['b']", "add_tag('c', ['x']) == ['x', 'c']"],
     "def add_tag(tag, tags=None):\n    tags = [] if tags is None else tags\n    tags.append(tag)\n    return tags"),
    ("find_max", "def find_max(nums):\n    best = 0\n    for n in nums:\n        if n > best:\n            best = n\n    return best",
     "find_max([-5, -2, -9]) returns 0 but should return -2.", ["find_max([-5, -2, -9]) == -2", "find_max([1, 7, 3]) == 7",
                                                               "find_max([4]) == 4"],
     "def find_max(nums):\n    return max(nums)"),
    ("average", "def average(nums):\n    return sum(nums) // len(nums)",
     "average([1, 2]) returns 1 but should return 1.5, and average([]) should return 0.0 instead of raising.",
     ["average([1, 2]) == 1.5", "average([2, 4, 6]) == 4.0", "average([]) == 0.0"],
     "def average(nums):\n    return sum(nums) / len(nums) if nums else 0.0"),
    ("is_palindrome", "def is_palindrome(s):\n    return s == s[::-1]",
     "It should ignore case and any non-alphanumeric characters: is_palindrome('A man, a plan, a canal: Panama') "
     "returns False but should return True.",
     ["is_palindrome('A man, a plan, a canal: Panama') is True", "is_palindrome('race a car') is False",
      "is_palindrome('') is True", "is_palindrome(\"No 'x' in Nixon\") is True"],
     "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]"),
    ("fib", "def fib(n):\n    if n <= 1:\n        return 1\n    return fib(n - 1) + fib(n - 2)",
     "fib(0) returns 1 but should return 0, and fib(10) returns 89 but should return 55.",
     ["fib(0) == 0", "fib(1) == 1", "fib(2) == 1", "fib(10) == 55"],
     "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"),
    ("word_counts", "def word_counts(text):\n    counts = {}\n    for word in text.split():\n"
     "        counts[word] = counts.get(word, 1) + 1\n    return counts",
     "word_counts('a b a') returns {'a': 3, 'b': 2} but should return {'a': 2, 'b': 1}.",
     ["word_counts('a b a') == {'a': 2, 'b': 1}", "word_counts('') == {}"],
     "def word_counts(text):\n    counts = {}\n    for w in text.split():\n        counts[w] = counts.get(w, 0) + 1\n    return counts"),
    ("sort_by_score", "def sort_by_score(pairs):\n    \"\"\"pairs: a list of (name, score); highest score first, ties by name A-Z.\"\"\"\n"
     "    return sorted(pairs, key=lambda p: p[1])", "The lowest score comes first, and ties are not ordered by name.",
     ["sort_by_score([('bo', 3), ('al', 5), ('cy', 3)]) == [('al', 5), ('bo', 3), ('cy', 3)]", "sort_by_score([]) == []"],
     "def sort_by_score(pairs):\n    return sorted(pairs, key=lambda p: (-p[1], p[0]))"),
]

MATH = [  # (question, answer)
    ("A shop sells pens at $3 each and notebooks at $7 each. Maya buys 4 pens and 3 notebooks and pays with a $50 "
     "bill. How much change, in dollars, does she get?", 17),
    ("A train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed in km/h over the whole "
     "trip?", 75),
    ("A tank holds 1,200 litres and is 35% full. How many more litres are needed to fill it?", 780),
    ("Three friends split a $96 bill so that Ana pays twice what Ben pays, and Ben pays the same as Cal. How many "
     "dollars does Ana pay?", 48),
    ("A rectangle's length is 3 times its width and its perimeter is 64 cm. What is its area in square cm?", 192),
    ("How many integers from 1 to 100 inclusive are divisible by 3 or by 5?", 47),
    ("A recipe needs 250 g of flour for 10 cookies. How many grams of flour are needed for 36 cookies?", 900),
    ("Lena saves $15 in the first week and each week saves $5 more than the week before. How many dollars has she "
     "saved in total after 8 weeks?", 260),
    ("What is the smaller angle, in degrees, between the hour and minute hands of a clock at 3:30?", 75),
    ("If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines take to make 100 widgets?", 5),
    ("The product of two consecutive positive integers is 342. What is the larger of the two?", 19),
    ("A car loses 20% of its value each year. It costs $25,000 new. What is it worth, in dollars, after 2 years?", 16000),
]

JSON = [  # (text, keys description, expected)
    ("Invoice INV-2048 was issued to Northwind Traders on 2026-03-14 for a total of 1,250.50 EUR, due in 30 days.",
     'invoice_id (string), customer (string), date (string, YYYY-MM-DD), total (number), currency (string), '
     'due_days (integer)',
     {"invoice_id": "INV-2048", "customer": "Northwind Traders", "date": "2026-03-14", "total": 1250.5,
      "currency": "EUR", "due_days": 30}),
    ("The meeting with Priya Shah and Tom Lee moved from Tuesday to Thursday at 15:30 in room B12.",
     'attendees (list of strings, in the order mentioned), day (string), time (string, HH:MM), room (string)',
     {"attendees": ["Priya Shah", "Tom Lee"], "day": "Thursday", "time": "15:30", "room": "B12"}),
    ("Order #7781: 3 x USB-C cable at $9.99, 1 x charger at $24.50. Ship to Austin, TX.",
     'order_id (integer), items (list of objects with name (string), qty (integer), price (number), in order), '
     'city (string), state (string, two letters)',
     {"order_id": 7781, "items": [{"name": "USB-C cable", "qty": 3, "price": 9.99},
                                  {"name": "charger", "qty": 1, "price": 24.5}], "city": "Austin", "state": "TX"}),
    ("Error 503 on api.example.com at 02:14 UTC; retried 4 times; resolved after failover to us-east-2.",
     'status (integer), host (string), time (string, HH:MM), retries (integer), region (string)',
     {"status": 503, "host": "api.example.com", "time": "02:14", "retries": 4, "region": "us-east-2"}),
    ("Flight BA 117 from London Heathrow to New York JFK departs at 08:45 on 2026-07-02 from gate C54; seat 23A.",
     'flight (string), origin (string), destination (string), departure (string, HH:MM), date (string, '
     'YYYY-MM-DD), gate (string), seat (string)',
     {"flight": "BA 117", "origin": "London Heathrow", "destination": "New York JFK", "departure": "08:45",
      "date": "2026-07-02", "gate": "C54", "seat": "23A"}),
    ("In May, Ravi paid 1200 for rent, 340 for groceries and 85 for internet.",
     'month (string), expenses (object with rent, groceries and internet as numbers), total (number, their sum)',
     {"month": "May", "expenses": {"rent": 1200, "groceries": 340, "internet": 85}, "total": 1625}),
]

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                    "required": ["city", "unit"]}}},
    {"type": "function", "function": {"name": "convert_currency", "description": "Convert an amount between currencies.",
     "parameters": {"type": "object", "properties": {"amount": {"type": "number"},
                                                    "from_currency": {"type": "string", "description": "ISO 4217 code"},
                                                    "to_currency": {"type": "string", "description": "ISO 4217 code"}},
                    "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "create_calendar_event", "description": "Create a calendar event.",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"},
                                                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                                                    "time": {"type": "string", "description": "HH:MM, 24-hour"},
                                                    "duration_minutes": {"type": "integer"}},
                    "required": ["title", "date", "time", "duration_minutes"]}}},
    {"type": "function", "function": {"name": "search_files", "description": "Search the user's files.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                                                    "extension": {"type": "string", "description": "file extension, e.g. py"}},
                    "required": ["query"]}}},
]

TOOL_CALLS = [  # (request, expected tool, argument check)
    ("What's the weather in Tokyo right now, in fahrenheit?", "get_weather",
     lambda a: str(a.get("city", "")).lower().startswith("tokyo") and a.get("unit") == "fahrenheit"),
    ("How much is 250 US dollars in euros?", "convert_currency",
     lambda a: float(a.get("amount", 0)) == 250 and str(a.get("from_currency")).upper() == "USD"
     and str(a.get("to_currency")).upper() == "EUR"),
    ("Book a 45-minute call titled 'Design review' on 2026-12-03 at 14:00.", "create_calendar_event",
     lambda a: a.get("title") == "Design review" and a.get("date") == "2026-12-03" and a.get("time") == "14:00"
     and int(a.get("duration_minutes", 0)) == 45),
    ("Find my Python files about tokenizers.", "search_files",
     lambda a: "tokenizer" in str(a.get("query", "")).lower() and str(a.get("extension", "")).lstrip(".").lower() == "py"),
    ("Is it raining in Paris? Use celsius.", "get_weather",
     lambda a: str(a.get("city", "")).lower().startswith("paris") and a.get("unit") == "celsius"),
    ("Convert 1,000 Japanese yen to British pounds.", "convert_currency",
     lambda a: float(a.get("amount", 0)) == 1000 and str(a.get("from_currency")).upper() == "JPY"
     and str(a.get("to_currency")).upper() == "GBP"),
]


def tasks():
    out = []
    for name, spec, tests, ref in CODE:
        out.append(dict(id=f"code/{name}", cat="write code", max_tokens=700, tests=tests, reference=f"```python\n{ref}\n```",
                        messages=[{"role": "user", "content": f"Write Python: {spec}\nReply with only the code, in one "
                                                              "```python block."}]))
    for name, buggy, symptom, tests, ref in BUGS:
        out.append(dict(id=f"bug/{name}", cat="fix a bug", max_tokens=600, tests=tests, reference=f"```python\n{ref}\n```",
                        messages=[{"role": "user", "content": f"This Python function has a bug.\n\n```python\n{buggy}\n```"
                                                              f"\n\n{symptom}\nReply with only the corrected function, in "
                                                              "one ```python block."}]))
    for i, (q, ans) in enumerate(MATH):
        out.append(dict(id=f"math/{i}", cat="word problem", max_tokens=600, answer=ans, reference=f"Answer: {ans}",
                        messages=[{"role": "user", "content": f"{q}\nWork it out briefly, then end with a line "
                                                              "'Answer: <number>'."}]))
    import json
    for i, (text, keys, exp) in enumerate(JSON):
        out.append(dict(id=f"json/{i}", cat="extract JSON", max_tokens=300, expected=exp, reference=json.dumps(exp),
                        messages=[{"role": "user", "content": f"Text: {text}\n\nExtract these fields: {keys}. Reply with "
                                                              "only a JSON object with exactly these keys."}]))
    for i, (req, tool, check) in enumerate(TOOL_CALLS):
        out.append(dict(id=f"tool/{i}", cat="tool call", max_tokens=300, tool=tool, check=check, tools=TOOLS,
                        messages=[{"role": "user", "content": req}]))
    return out


# ------------------------------------------------------------------ the harder tier (the first set left no gaps)
HARD_CODE = [
    ("edit_distance", "edit_distance(a, b) returns the Levenshtein distance between strings a and b (insertions, "
     "deletions and substitutions each cost 1).",
     ["edit_distance('kitten','sitting') == 3", "edit_distance('','abc') == 3", "edit_distance('flaw','lawn') == 2",
      "edit_distance('same','same') == 0"],
     "def edit_distance(a, b):\n    d = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n        p, d[0] = d[0], i\n"
     "        for j, cb in enumerate(b, 1):\n            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))\n    return d[-1]"),
    ("regex_match", "regex_match(s, p) returns True when the pattern p matches the whole string s, where p has letters, "
     "'.' (any one character) and '*' (zero or more of the preceding element); no other regex features.",
     ["regex_match('aa','a') is False", "regex_match('aa','a*') is True", "regex_match('ab','.*') is True",
      "regex_match('aab','c*a*b') is True", "regex_match('mississippi','mis*is*p*.') is False"],
     "def regex_match(s, p):\n    from functools import lru_cache\n    @lru_cache(None)\n    def m(i, j):\n"
     "        if j == len(p):\n            return i == len(s)\n        first = i < len(s) and p[j] in (s[i], '.')\n"
     "        if j + 1 < len(p) and p[j + 1] == '*':\n            return m(i, j + 2) or (first and m(i + 1, j))\n"
     "        return first and m(i + 1, j + 1)\n    return m(0, 0)"),
    ("topo_sort", "topo_sort(n, edges) returns an ordering of the nodes 0..n-1 in which every edge (u, v) has u before "
     "v, or [] when the edges contain a cycle.",
     ["(lambda o, E: sorted(o) == list(range(4)) and all(o.index(u) < o.index(v) for u, v in E))"
      "(topo_sort(4, [(0,1),(1,2),(0,3)]), [(0,1),(1,2),(0,3)])", "topo_sort(3, [(0,1),(1,2),(2,0)]) == []",
      "sorted(topo_sort(2, [])) == [0, 1]"],
     "def topo_sort(n, edges):\n    indeg = [0] * n\n    adj = [[] for _ in range(n)]\n    for u, v in edges:\n"
     "        adj[u].append(v)\n        indeg[v] += 1\n    q = [i for i in range(n) if indeg[i] == 0]\n    out = []\n"
     "    while q:\n        u = q.pop()\n        out.append(u)\n        for v in adj[u]:\n            indeg[v] -= 1\n"
     "            if indeg[v] == 0:\n                q.append(v)\n    return out if len(out) == n else []"),
    ("lis_length", "lis_length(nums) returns the length of the longest strictly increasing subsequence of nums.",
     ["lis_length([10,9,2,5,3,7,101,18]) == 4", "lis_length([]) == 0", "lis_length([7,7,7]) == 1",
      "lis_length([0,1,0,3,2,3]) == 4"],
     "def lis_length(nums):\n    import bisect\n    t = []\n    for x in nums:\n        i = bisect.bisect_left(t, x)\n"
     "        t[i:i + 1] = [x]\n    return len(t)"),
    ("eval_expr", "eval_expr(s) evaluates an arithmetic expression string with numbers, +, -, *, / (true division), "
     "unary minus and parentheses, honouring precedence, and returns the value. Do not use eval or exec.",
     ["eval_expr('2+3*4') == 14", "eval_expr('(2+3)*4') == 20", "eval_expr('10/4') == 2.5", "eval_expr('-3+5') == 2",
      "eval_expr('2*(3+(4-1))/3') == 4"],
     "def eval_expr(s):\n    import re\n    toks = re.findall(r'\\d+\\.?\\d*|[-+*/()]', s)\n    pos = [0]\n"
     "    def peek():\n        return toks[pos[0]] if pos[0] < len(toks) else None\n    def take():\n        pos[0] += 1\n        return toks[pos[0] - 1]\n"
     "    def expr():\n        v = term()\n        while peek() in ('+', '-'):\n            v = v + term() if take() == '+' else v - term()\n        return v\n"
     "    def term():\n        v = unary()\n        while peek() in ('*', '/'):\n            v = v * unary() if take() == '*' else v / unary()\n        return v\n"
     "    def unary():\n        if peek() == '-':\n            take()\n            return -unary()\n        if peek() == '(':\n            take()\n"
     "            v = expr()\n            take()\n            return v\n        return float(take())\n    return expr()"),
    ("min_meeting_rooms", "min_meeting_rooms(intervals) returns the minimum number of rooms needed for meetings given "
     "as [start, end] (a meeting ending at t frees its room for one starting at t).",
     ["min_meeting_rooms([[0,30],[5,10],[15,20]]) == 2", "min_meeting_rooms([[7,10],[2,4]]) == 1",
      "min_meeting_rooms([]) == 0", "min_meeting_rooms([[1,5],[5,10]]) == 1"],
     "def min_meeting_rooms(intervals):\n    ev = sorted([(s, 1) for s, e in intervals] + [(e, -1) for s, e in intervals])\n"
     "    cur = best = 0\n    for _, d in ev:\n        cur += d\n        best = max(best, cur)\n    return best"),
    ("word_break", "word_break(s, words) returns True when the string s can be split into a sequence of one or more "
     "words from the list words (words may repeat).",
     ["word_break('leetcode', ['leet','code']) is True", "word_break('applepenapple', ['apple','pen']) is True",
      "word_break('catsandog', ['cats','dog','sand','and','cat']) is False"],
     "def word_break(s, words):\n    ok = [True] + [False] * len(s)\n    for i in range(1, len(s) + 1):\n"
     "        ok[i] = any(ok[i - len(w)] and s[i - len(w):i] == w for w in words if len(w) <= i)\n    return ok[-1]"),
    ("spiral_order", "spiral_order(matrix) returns the elements of the matrix (a list of equal-length rows) in clockwise "
     "spiral order starting at the top-left.",
     ["spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]",
      "spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]", "spiral_order([]) == []"],
     "def spiral_order(matrix):\n    out, m = [], [list(r) for r in matrix]\n    while m:\n        out += m.pop(0)\n"
     "        m = [list(r) for r in zip(*m)][::-1]\n    return out"),
    ("decode_string", "decode_string(s) expands strings like '3[a2[c]]' (k[...] repeats the bracketed part k times, "
     "nesting allowed) and returns the result.",
     ["decode_string('3[a2[c]]') == 'accaccacc'", "decode_string('2[abc]3[cd]ef') == 'abcabccdcdcdef'",
      "decode_string('abc') == 'abc'"],
     "def decode_string(s):\n    st, cur, k = [], '', 0\n    for c in s:\n        if c.isdigit():\n            k = k * 10 + int(c)\n"
     "        elif c == '[':\n            st.append((cur, k))\n            cur, k = '', 0\n        elif c == ']':\n"
     "            prev, n = st.pop()\n            cur = prev + cur * n\n        else:\n            cur += c\n    return cur"),
    ("next_permutation", "next_permutation(nums) returns, as a new list, the next lexicographically greater permutation "
     "of nums, or the lowest (sorted) permutation when nums is the highest.",
     ["next_permutation([1,2,3]) == [1,3,2]", "next_permutation([3,2,1]) == [1,2,3]", "next_permutation([1,1,5]) == [1,5,1]",
      "next_permutation([1,3,2]) == [2,1,3]"],
     "def next_permutation(nums):\n    a = list(nums)\n    i = len(a) - 2\n    while i >= 0 and a[i] >= a[i + 1]:\n        i -= 1\n"
     "    if i >= 0:\n        j = len(a) - 1\n        while a[j] <= a[i]:\n            j -= 1\n        a[i], a[j] = a[j], a[i]\n"
     "    a[i + 1:] = reversed(a[i + 1:])\n    return a"),
]

HARD_BUGS = [
    ("bsearch", "def bsearch(a, x):\n    lo, hi = 0, len(a) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n"
     "        if a[mid] < x:\n            lo = mid\n        else:\n            hi = mid\n    return lo if a and a[lo] == x else -1",
     "bsearch([1, 3], 3) never returns. It should return the index of x in the sorted list a, or -1 when absent.",
     ["bsearch([1, 3], 3) == 1", "bsearch([1, 3, 5, 7], 5) == 2", "bsearch([1, 3], 2) == -1", "bsearch([], 1) == -1"],
     "def bsearch(a, x):\n    lo, hi = 0, len(a) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if a[mid] == x:\n"
     "            return mid\n        if a[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1"),
    ("make_multipliers", "def make_multipliers(n):\n    return [lambda x: x * i for i in range(n)]",
     "make_multipliers(3)[0](10) returns 20 but should return 0: each function should multiply by its own index.",
     ["[f(10) for f in make_multipliers(3)] == [0, 10, 20]", "make_multipliers(0) == []"],
     "def make_multipliers(n):\n    return [lambda x, i=i: x * i for i in range(n)]"),
    ("drop_empty", "def drop_empty(d):\n    for k in d:\n        if not d[k]:\n            del d[k]\n    return d",
     "drop_empty({'a': 1, 'b': 0, 'c': ''}) raises RuntimeError but should return {'a': 1}.",
     ["drop_empty({'a': 1, 'b': 0, 'c': ''}) == {'a': 1}", "drop_empty({}) == {}"],
     "def drop_empty(d):\n    for k in [k for k in d if not d[k]]:\n        del d[k]\n    return d"),
    ("split_bill", "def split_bill(total, people):\n    share = round(total / people, 2)\n    return [share] * people",
     "split_bill(100, 3) returns three shares of 33.33 that add up to 99.99. The shares must add up to the total exactly "
     "(to the cent), with any extra cents going to the first people.",
     ["split_bill(100, 3) == [33.34, 33.33, 33.33]", "split_bill(10, 2) == [5.0, 5.0]", "split_bill(0.05, 2) == [0.03, 0.02]"],
     "def split_bill(total, people):\n    cents = round(total * 100)\n    base, extra = divmod(cents, people)\n"
     "    return [(base + (1 if i < extra else 0)) / 100 for i in range(people)]"),
]

HARD_MATH = [
    ("How many positive divisors does 360 have?", 24),
    ("What is the remainder when 7^100 is divided by 13?", 9),
    ("In how many ways can 8 people sit around a round table, if seatings that differ only by rotation count as the "
     "same?", 5040),
    ("The sum of three consecutive odd integers is 159. What is the largest of them?", 55),
    ("A bag holds 5 red and 3 blue balls. Two are drawn without replacement. The probability that both are red is a/b in "
     "lowest terms. What is a + b?", 19),
    ("What is the sum of all positive integers n below 50 for which n^2 + 1 is divisible by 5?", 500),
    ("$10,000 earns 6% interest a year, compounded yearly. What is the balance after 3 years, to the nearest dollar?",
     11910),
    ("In how many ways can you make 25 cents from pennies, nickels and dimes?", 12),
    ("Two pipes fill a pool alone in 6 hours and 3 hours; a drain empties the full pool in 4 hours. With all three open, "
     "how many hours does it take to fill the empty pool?", 4),
    ("What are the last two digits of 3^20, read as a number?", 1),
]

HARD_JSON = [
    ("The shipment from Rotterdam left on the 3rd of March 2026 carrying twelve containers, each weighing 18.5 tonnes.",
     'origin (string), date (string, YYYY-MM-DD), containers (integer), total_tonnes (number: all containers together)',
     {"origin": "Rotterdam", "date": "2026-03-03", "containers": 12, "total_tonnes": 222}),
    ("Sam (sam@ex.io) and Jo (jo@ex.io) are admins; Lee (lee@ex.io) is a viewer.",
     'admins (list of email strings, in order), viewers (list of email strings)',
     {"admins": ["sam@ex.io", "jo@ex.io"], "viewers": ["lee@ex.io"]}),
    ("The price rose from $80 to $92 over the quarter.",
     'start (number), end (number), change_percent (number, the percentage change)',
     {"start": 80, "end": 92, "change_percent": 15}),
    ("Doors open at 7pm; the show starts 45 minutes later and runs for 2 hours 15 minutes.",
     'doors (string, HH:MM 24-hour), start (string, HH:MM), end (string, HH:MM)',
     {"doors": "19:00", "start": "19:45", "end": "22:00"}),
]

HARD_TOOL_CALLS = [
    ("I'm flying to Berlin next week. How warm is it there right now, in celsius?", "get_weather",
     lambda a: str(a.get("city", "")).lower().startswith("berlin") and a.get("unit") == "celsius"),
    ("How much is 99.5 Swiss francs in yen?", "convert_currency",
     lambda a: float(a.get("amount", 0)) == 99.5 and str(a.get("from_currency")).upper() == "CHF"
     and str(a.get("to_currency")).upper() == "JPY"),
    ("Set up a 90 minute 'Quarterly planning' session on 2027-01-15 starting at 9am.", "create_calendar_event",
     lambda a: a.get("title") == "Quarterly planning" and a.get("date") == "2027-01-15" and a.get("time") == "09:00"
     and int(a.get("duration_minutes", 0)) == 90),
    ("Look for my markdown notes that mention the roadmap.", "search_files",
     lambda a: "roadmap" in str(a.get("query", "")).lower() and str(a.get("extension", "")).lstrip(".").lower() == "md"),
]


def hard_tasks():
    import json
    out = []
    for name, spec, tests, ref in HARD_CODE:
        out.append(dict(id=f"hard/code/{name}", cat="write code", max_tokens=1000, tests=tests,
                        reference=f"```python\n{ref}\n```",
                        messages=[{"role": "user", "content": f"Write Python: {spec}\nReply with only the code, in one "
                                                              "```python block."}]))
    for name, buggy, symptom, tests, ref in HARD_BUGS:
        out.append(dict(id=f"hard/bug/{name}", cat="fix a bug", max_tokens=800, tests=tests,
                        reference=f"```python\n{ref}\n```",
                        messages=[{"role": "user", "content": f"This Python function has a bug.\n\n```python\n{buggy}\n```"
                                                              f"\n\n{symptom}\nReply with only the corrected function, in "
                                                              "one ```python block."}]))
    for i, (q, ans) in enumerate(HARD_MATH):
        out.append(dict(id=f"hard/math/{i}", cat="word problem", max_tokens=1000, answer=ans, reference=f"Answer: {ans}",
                        messages=[{"role": "user", "content": f"{q}\nWork it out briefly, then end with a line "
                                                              "'Answer: <number>'."}]))
    for i, (text, keys, exp) in enumerate(HARD_JSON):
        out.append(dict(id=f"hard/json/{i}", cat="extract JSON", max_tokens=400, expected=exp, reference=json.dumps(exp),
                        messages=[{"role": "user", "content": f"Text: {text}\n\nExtract these fields: {keys}. Reply with "
                                                              "only a JSON object with exactly these keys."}]))
    for i, (req, tool, check) in enumerate(HARD_TOOL_CALLS):
        out.append(dict(id=f"hard/tool/{i}", cat="tool call", max_tokens=400, tool=tool, check=check, tools=TOOLS,
                        messages=[{"role": "user", "content": req}]))
    return out
