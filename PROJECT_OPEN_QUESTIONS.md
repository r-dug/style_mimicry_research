# Project Open Questions

These are the main planning questions to resolve before coding implementation details.

## Path and Naming Conventions

1. 
   - *Q:* Should `data/style_mimicry/` be the canonical project data root? (`Project_PLAN.md` still references `./i_o/style_mimicry_res` and `i_o/style_mimicry/original_art`.)
   - *A:* Yes. You may modify the plan to reflect this change.
2. 
   - *Q:* Do you want one naming standard across scripts and data folders for methods?  
   Current mismatches include:
      - `diffPure` (data) vs `diff_pure` (code folder)
      - `gauss_noise` (data) vs `gaussian_noise` (code folder)
      - `noisy_upscale` (data) vs `noisy_upscaling` (code folder)
      - `impress` (folder) vs `Impress++` (plan name)
   - *A:* Yes. Please update mismatches to match:
      - `diff_pure`
      - `gaussian_noise`
      - `noisy_upscaling`
      - `impress`
3. 
   - *Q:* Should protection names be normalized as one convention (all caps or snake_case) across `mimicry/protections` and `data/style_mimicry/protected_art`?
   - *A:* Style notes:
      - for directories, you can use snake_case. 
      - In general, follow [pep style convention](https://peps.python.org/pep-0008/) for python.
      - Pleae use concise but informative doc strings 
      - Use type hinting for class and function parameters and returns, as well as constants.

## Workflow Scope and Deliverables

1. 
   - *Q:* Is `mimicry/protections/pull_protections.sh` intended to exist? (It is open in the IDE tab list but not present on disk.)
   - *A:* I changed the name to 'get_protections.sh' 
2. 
   - *Q:* For Step 3, what exact SD3.5 target(s) should we standardize on (e.g., SD3.5 Medium vs Large), and which training stack should be primary?
   - *A:* Lets use Large as the standard target. it is the most performant.
      - Let's use two training stacks: SimpleTuner and train_dreambooth_lora_sd3.py from hugging face diffusers.
      - We can try style transfer on both of them for a few experiments and decide which to run the entire experiment with later.
3. 
   - *Q:* Should generated model artifacts be uploaded automatically to Hugging Face during runs, or only manually after validation?
   - *A:* Automatically
4. 
   - *Q:* What should be the canonical location for experiment configs/parameter dictionaries (JSON/YAML/Python), since this is central to scenario coverage?
   - *A:* Python, for simplicity, unless you can think of a convincing reason to use another approach.

## Dataset and Experiment Design

1. 
   - *Q:* Do we proceed now with historical artists only, since contemporary artist folders are present but currently have no artist subfolders/images?
   - *A:* We can proceed with the historical artists we currently have. For every step that we implement, we should implement getter functions 
2. 
   - *Q:* What is the minimum per-artist sample threshold to proceed with training (strict 18-20 only, or allow exceptions)?
   - *A:* 20. We should not allow exceptions. You may modify the scraper script and run it to rectify this descrepency. 
3. 
   - *Q:* What image preprocessing standard should be fixed up front (resolution, aspect handling, file normalization)?
   - *A:* All of the above

## Evaluation and Analysis

1. 
   - *Q:* Which benchmark metrics should be treated as required for Step 5 (style similarity, CLIP-based metrics, image quality metrics, etc.)?
   - *A:* ise the benchmarking metrics suggested [in this paper](https://arxiv.org/pdf/2507.03953). You may use code from their open-source repo, [here](https://github.com/vkeilo/DiffAdvPerturbationBench)
2. 
   - *Q:* For survey design, do you want a predefined protocol now (sample size, randomization, evaluation rubric), or should we defer until first model outputs are available?
   - *A:* defer until first model outputs are available
3. 
   - *Q:* `Project_PLAN.md` jumps from Step 6 to Step 8. Should Step 7 be explicitly defined (e.g., data cleaning/aggregation before final analysis)?
   - *A:* Typo. There are currently only 7 conceptualized steps.


### Additional notes

- Reuse code from linked github repos, to save yourself coding effort.
- ask questions when unsure
- make sure that every step in this pipeline does book marking, making note of what has been done.
   - we want to track current progress and ensure completeness (comprehensive testing)
   - we want to prevent unnecessary computation and reduing work we have already done.
      - therefore, checks should be performed on the current state of the project, to determine what has been done, and what still must be done.
      - This progress tracking should be flexible enough to adjust to changes in the artists, original data, etc.

## CODEX Follow-up Questions

I reviewed your answers and they’re clear. I only have these follow-up questions before implementation:

1. 
- *Q:* For naming normalization, should I rename both code and existing data directories now (e.g., `diffPure` -> `diff_pure`, `Mist` -> `mist`), or keep current data dirs and normalize only in new code paths?
- *A:* All data and code directory names should match
2. 
- *Q:* For protection directory names, do you want this exact canonical set: `glaze`, `mist`, `anti_dreambooth`, `style_guard`?
- *A:* Yes
3. 
- *Q:* With the strict 20-image rule, what should happen if an artist cannot reach 20 after scraping retries: drop that artist, or replace with another artist in the same style?
- *A:* Log this in the book marking (track how many samples there are per artist as one progress metric)
4. 
- *Q:* Current counts show 11 historical artists below 20 (including `Rene_Magritte` at 10). Should I treat fixing these as the first implementation task?
- *A:* Sure. You would only need to slightly modify the scraper script
5. 
- *Q:* For preprocessing, what exact training resolution should we standardize on for SD3.5 Large (e.g., `1024x1024`), and should aspect handling be center-crop or pad?
- *A:* set a flag for resolution and aspect handling. the defaults can be (1024,1024) and 'center-crop', but be sure to implement a padding option too.
6. 
- *Q:* For the two-stack pilot (SimpleTuner vs `train_dreambooth_lora_sd3.py`), how many pilot artists/scenarios do you want before choosing one for full runs?
- *A:* one per style. you can consider this a 'trial' run for any given step of the pipeline.
7. 
- *Q:* For automatic Hugging Face upload, should model repos be private by default, and do you already have a target namespace/org to use?
- *A:* Yes, they should be private by default. I have not set up a target namespace/org yet but we can go through that together later.
8. 
- *Q:* For benchmarking, should I pin to a specific commit/version of `DiffAdvPerturbationBench` for reproducibility?
- *A:* yes. check for most current and make note of this in experimental setup. Oh... yeah. that should probably be included in a readme.md.
9. 
- *Q:* For “bookmarking/progress tracking,” do you want a single machine-readable tracker (JSON/SQLite) as source-of-truth, plus a human-readable summary log?
- *A:* Yes. JSON please.
10. 
- *Q:* Do you want me to update `Project_PLAN.md` now to reflect your decisions (path updates, naming standard, and step numbering fix to 7 steps)?
- *A:* Yes.