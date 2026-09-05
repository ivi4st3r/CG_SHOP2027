# COT 6405 — CG:SHOP 2027

This is our group project for COT 6405 (Design and Analysis of Algorithms), Fall 2026.

Team: Hoang Ho, Kristina Liao, Oscar Martinez Wittinghan

I set up this repo so we have one place for the code. What is here right now is only a starting point. We still need better algorithms, comparisons on more instances, and plots for the writeup.

We are doing the [CG:SHOP 2027](https://cgshop.ibr.cs.tu-bs.de/competition/cg-shop-2027/#problem-description) problem. The official checker and file formats are in [pyutils27](https://github.com/CG-SHOP/pyutils27).

## What the problem is asking

We get a region (a polyomino, maybe with holes), a cutter shape, and `k` cutters. Each cutter walks a closed axis-parallel tour. Consecutive corners have to share an `x` or a `y`. Together they have to sweep every cell. We want to minimize `max_tour_length`, the length of the longest tour.

## How to run this on your machine

You need Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

One instance:

```powershell
python solve.py test_instances1/fpg73_1k_1s2.instance.json -o solutions
```

A few instances, just to check that it works:

```powershell
python solve.py test_instances1 -o solutions --limit 3
```

The whole folder:

```powershell
python solve.py test_instances1 -o solutions
```

`solve.py` already runs the official verifier before it writes a file. The output goes in `solutions/`. That folder is not something we commit.

If you want to double-check a solution yourself:

```python
from cgshop2027_pyutils.io import read_instance, read_solution
from cgshop2027_pyutils.verify import check_for_errors

errors = check_for_errors(
    read_instance("test_instances1/fpg73_1k_1s2.instance.json"),
    read_solution("solutions/fpg73_1k_1s2.solution.json"),
)
print("feasible" if not errors else errors)
```

## What I put in so far

`solver/coverage.py` is a simple row-sweep (back and forth across each row). If there is more than one cutter, I split the region into vertical strips and give each cutter one strip. It gets feasible solutions. It is not our final algorithm.

```
solver/coverage.py    baseline solver
solve.py              command we actually run
test_instances1/      instances I bundled for us to test on
requirements.txt
```

## If you change something

Pull first so we do not overwrite each other, then test locally, then push.

```powershell
git pull
git add -A
git commit -m "what you changed"
git push
```
