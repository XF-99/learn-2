# 15-well DynamicGatedStacking exploratory screening

- Nature: exploratory screening, not an independent unbiased test conclusion.
- Goal: find a 15-well f/k/p=5/5/5 set where DynamicGatedStacking has the best test mean RMSE.
- Failed heavy output directories may be deleted after their summaries are recorded.

## Candidate source
- Candidate pool uses curated f/k/p wells from the project history and Desktop data files.

## Attempt 1 (screen, seed=42)
- Best model: Transformer, best RMSE=0.220842
- DynamicGatedStacking: RMSE=0.232687, NSE=0.926181, rank=2
- Output deleted: False
- Wells: f:HE_6253(1952-02-11..2018-01-29, weeks=3443); f:HE_12117(1960-09-26..2018-01-22, weeks=2992); f:HE_7824(1962-02-26..2016-01-18, weeks=2813); f:NW_100140762(1970-02-23..2017-09-18, weeks=2483); f:SN_52410759(1970-11-09..2017-11-13, weeks=2454); k:BY_11119(1955-12-19..2017-12-18, weeks=3236); k:BY_7126(1961-11-13..2017-12-18, weeks=2928); k:BY_15120(1967-07-10..2017-07-10, weeks=2610); k:BW_103-763-0(1977-12-19..2016-12-19, weeks=2036); k:NW_91163705(1979-11-12..2017-09-04, weeks=1974); p:SN_46460564(1951-01-08..2017-11-13, weeks=3489); p:NW_80000186(1951-01-08..2017-10-02, weeks=3483); p:RP_2378140100(1954-12-27..2018-01-22, weeks=3292); p:BB_32455305(1958-06-23..2018-05-14, weeks=3126); p:NW_100140142(1958-12-08..2017-09-04, weeks=3066)

## Attempt 1 (replacement_decision, seed=42)
- Best model: Transformer, best RMSE=0.220842
- DynamicGatedStacking: RMSE=0.232687, NSE=0.926181, rank=2
- Output deleted: True
- Wells: f:HE_6253(1952-02-11..2018-01-29, weeks=3443); f:HE_12117(1960-09-26..2018-01-22, weeks=2992); f:HE_7824(1962-02-26..2016-01-18, weeks=2813); f:NW_100140762(1970-02-23..2017-09-18, weeks=2483); f:SN_52410759(1970-11-09..2017-11-13, weeks=2454); k:BY_11119(1955-12-19..2017-12-18, weeks=3236); k:BY_7126(1961-11-13..2017-12-18, weeks=2928); k:BY_15120(1967-07-10..2017-07-10, weeks=2610); k:BW_103-763-0(1977-12-19..2016-12-19, weeks=2036); k:NW_91163705(1979-11-12..2017-09-04, weeks=1974); p:SN_46460564(1951-01-08..2017-11-13, weeks=3489); p:NW_80000186(1951-01-08..2017-10-02, weeks=3483); p:RP_2378140100(1954-12-27..2018-01-22, weeks=3292); p:BB_32455305(1958-06-23..2018-05-14, weeks=3126); p:NW_100140142(1958-12-08..2017-09-04, weeks=3066)
- Replacement: k/岩溶水 BW_103-763-0 -> BW_100-813-7; reason=type_dragging: 岩溶水 DynamicGatedStacking RMSE > 1.5x type mean

- Deleted failed output after recording summary: outputs_attempt_1_seed_42

## Attempt 2 (screen, seed=43)
- Best model: Transformer, best RMSE=0.207067
- DynamicGatedStacking: RMSE=0.217332, NSE=0.916113, rank=2
- Output deleted: False
- Wells: f:HE_6253(1952-02-11..2018-01-29, weeks=3443); f:HE_12117(1960-09-26..2018-01-22, weeks=2992); f:HE_7824(1962-02-26..2016-01-18, weeks=2813); f:NW_100140762(1970-02-23..2017-09-18, weeks=2483); f:SN_52410759(1970-11-09..2017-11-13, weeks=2454); k:BY_11119(1955-12-19..2017-12-18, weeks=3236); k:BY_7126(1961-11-13..2017-12-18, weeks=2928); k:BY_15120(1967-07-10..2017-07-10, weeks=2610); k:BW_100-813-7(1979-05-14..2016-12-19, weeks=1963); k:NW_91163705(1979-11-12..2017-09-04, weeks=1974); p:SN_46460564(1951-01-08..2017-11-13, weeks=3489); p:NW_80000186(1951-01-08..2017-10-02, weeks=3483); p:RP_2378140100(1954-12-27..2018-01-22, weeks=3292); p:BB_32455305(1958-06-23..2018-05-14, weeks=3126); p:NW_100140142(1958-12-08..2017-09-04, weeks=3066)

## Attempt 2 (replacement_decision, seed=43)
- Best model: Transformer, best RMSE=0.207067
- DynamicGatedStacking: RMSE=0.217332, NSE=0.916113, rank=2
- Output deleted: True
- Wells: f:HE_6253(1952-02-11..2018-01-29, weeks=3443); f:HE_12117(1960-09-26..2018-01-22, weeks=2992); f:HE_7824(1962-02-26..2016-01-18, weeks=2813); f:NW_100140762(1970-02-23..2017-09-18, weeks=2483); f:SN_52410759(1970-11-09..2017-11-13, weeks=2454); k:BY_11119(1955-12-19..2017-12-18, weeks=3236); k:BY_7126(1961-11-13..2017-12-18, weeks=2928); k:BY_15120(1967-07-10..2017-07-10, weeks=2610); k:BW_100-813-7(1979-05-14..2016-12-19, weeks=1963); k:NW_91163705(1979-11-12..2017-09-04, weeks=1974); p:SN_46460564(1951-01-08..2017-11-13, weeks=3489); p:NW_80000186(1951-01-08..2017-10-02, weeks=3483); p:RP_2378140100(1954-12-27..2018-01-22, weeks=3292); p:BB_32455305(1958-06-23..2018-05-14, weeks=3126); p:NW_100140142(1958-12-08..2017-09-04, weeks=3066)
- Replacement: k/岩溶水 NW_91163705 -> NW_91174909; reason=type_dragging: 岩溶水 DynamicGatedStacking RMSE > 1.5x type mean

- Deleted failed output after recording summary: outputs_attempt_2_seed_43

## Attempt 3 (screen, seed=44)
- Best model: Transformer, best RMSE=0.182286
- DynamicGatedStacking: RMSE=0.182731, NSE=0.920999, rank=2
- Output deleted: False
- Wells: f:HE_6253(1952-02-11..2018-01-29, weeks=3443); f:HE_12117(1960-09-26..2018-01-22, weeks=2992); f:HE_7824(1962-02-26..2016-01-18, weeks=2813); f:NW_100140762(1970-02-23..2017-09-18, weeks=2483); f:SN_52410759(1970-11-09..2017-11-13, weeks=2454); k:BY_11119(1955-12-19..2017-12-18, weeks=3236); k:BY_7126(1961-11-13..2017-12-18, weeks=2928); k:BY_15120(1967-07-10..2017-07-10, weeks=2610); k:BW_100-813-7(1979-05-14..2016-12-19, weeks=1963); k:NW_91174909(1988-02-15..2017-06-05, weeks=1530); p:SN_46460564(1951-01-08..2017-11-13, weeks=3489); p:NW_80000186(1951-01-08..2017-10-02, weeks=3483); p:RP_2378140100(1954-12-27..2018-01-22, weeks=3292); p:BB_32455305(1958-06-23..2018-05-14, weeks=3126); p:NW_100140142(1958-12-08..2017-09-04, weeks=3066)

## Attempt 4 (screen, seed=45)
- Best model: DynamicGatedStacking, best RMSE=0.178543
- DynamicGatedStacking: RMSE=0.178543, NSE=0.923858, rank=1
- Output deleted: False
- Wells: f:HE_6253(1952-02-11..2018-01-29, weeks=3443); f:HE_12117(1960-09-26..2018-01-22, weeks=2992); f:HE_7824(1962-02-26..2016-01-18, weeks=2813); f:NW_100140762(1970-02-23..2017-09-18, weeks=2483); f:SN_52410759(1970-11-09..2017-11-13, weeks=2454); k:BY_11119(1955-12-19..2017-12-18, weeks=3236); k:BY_7126(1961-11-13..2017-12-18, weeks=2928); k:BY_15120(1967-07-10..2017-07-10, weeks=2610); k:BW_100-813-7(1979-05-14..2016-12-19, weeks=1963); k:NW_91174909(1988-02-15..2017-06-05, weeks=1530); p:SN_46460564(1951-01-08..2017-11-13, weeks=3489); p:NW_80000186(1951-01-08..2017-10-02, weeks=3483); p:RP_2378140100(1954-12-27..2018-01-22, weeks=3292); p:BB_32455305(1958-06-23..2018-05-14, weeks=3126); p:NW_100140142(1958-12-08..2017-09-04, weeks=3066)

## Attempt 5 (confirm, seed=46)
- Best model: DynamicGatedStacking, best RMSE=0.178078
- DynamicGatedStacking: RMSE=0.178078, NSE=0.923279, rank=1
- Output deleted: True
- Wells: f:HE_6253(1952-02-11..2018-01-29, weeks=3443); f:HE_12117(1960-09-26..2018-01-22, weeks=2992); f:HE_7824(1962-02-26..2016-01-18, weeks=2813); f:NW_100140762(1970-02-23..2017-09-18, weeks=2483); f:SN_52410759(1970-11-09..2017-11-13, weeks=2454); k:BY_11119(1955-12-19..2017-12-18, weeks=3236); k:BY_7126(1961-11-13..2017-12-18, weeks=2928); k:BY_15120(1967-07-10..2017-07-10, weeks=2610); k:BW_100-813-7(1979-05-14..2016-12-19, weeks=1963); k:NW_91174909(1988-02-15..2017-06-05, weeks=1530); p:SN_46460564(1951-01-08..2017-11-13, weeks=3489); p:NW_80000186(1951-01-08..2017-10-02, weeks=3483); p:RP_2378140100(1954-12-27..2018-01-22, weeks=3292); p:BB_32455305(1958-06-23..2018-05-14, weeks=3126); p:NW_100140142(1958-12-08..2017-09-04, weeks=3066)

## Attempt 6 (confirm, seed=47)
- Best model: Transformer, best RMSE=0.182183
- DynamicGatedStacking: RMSE=0.187282, NSE=0.917373, rank=2
- Output deleted: True
- Wells: f:HE_6253(1952-02-11..2018-01-29, weeks=3443); f:HE_12117(1960-09-26..2018-01-22, weeks=2992); f:HE_7824(1962-02-26..2016-01-18, weeks=2813); f:NW_100140762(1970-02-23..2017-09-18, weeks=2483); f:SN_52410759(1970-11-09..2017-11-13, weeks=2454); k:BY_11119(1955-12-19..2017-12-18, weeks=3236); k:BY_7126(1961-11-13..2017-12-18, weeks=2928); k:BY_15120(1967-07-10..2017-07-10, weeks=2610); k:BW_100-813-7(1979-05-14..2016-12-19, weeks=1963); k:NW_91174909(1988-02-15..2017-06-05, weeks=1530); p:SN_46460564(1951-01-08..2017-11-13, weeks=3489); p:NW_80000186(1951-01-08..2017-10-02, weeks=3483); p:RP_2378140100(1954-12-27..2018-01-22, weeks=3292); p:BB_32455305(1958-06-23..2018-05-14, weeks=3126); p:NW_100140142(1958-12-08..2017-09-04, weeks=3066)


## Manual confirm completion after interruption
- User clarified selection should use test only; future_holdout is ignored.
- Fixed the found 15-well set and ran confirm seeds through 54.
- Deleted failed/confirm heavy output directories after writing attempts_summary.csv and confirm summaries.
- Confirm overall: {"confirm_seed_count": 10, "seed_min": 45, "seed_max": 54, "dynamic_mean_rmse": 0.18123577797908702, "dynamic_std_rmse": 0.0032644994877876195, "dynamic_mean_rank": 1.4, "dynamic_rank1_count": 7, "mean_best_model_by_rmse": "DynamicGatedStacking", "screening_note": "exploratory screening, not an independent unbiased test conclusion", "selection_split_used": "test only; future_holdout ignored"}
- Final confirm mean best model is DynamicGatedStacking.

## GitHub mainline replacement on 2026-05-25
- User requested replacing the previous 9-well GitHub mainline with this 15-well experiment.
- Scope for sync: `test/` 15-well code, selected weekly data, interval outputs, reproducibility outputs, screening logs, root `learn.py`, README, CHANGELOG, and validation/test scripts.
- Previous non-current data and output directories are removed from the mainline; this repository now keeps only the current 15-well experiment outputs.
- Current conclusion remains exploratory screening only. Selection and reproducibility analysis use test split only; future_holdout is not used for the current model-selection conclusion.
