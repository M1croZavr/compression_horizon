
set -x

PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Llama-3.2-1B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Llama-3.2-3B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-160m_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-410m_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-135M_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-360M_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-270m_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-1b-pt_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lr_0.1/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --midrule_indicies 2 5 8 \
  --show_labels --only_stat_table --tablefmt latex


PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.01/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_1.0/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_5.0/progressive_prefixes \
  --names_mapping "0.01,0.1,0.5,1.0,5.0" \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex


# Full Llama31-8B
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1*/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex

# Full pythia-1.4b
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b*/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex

# Full SmalLM2
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B*/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex

# Full Qwen3-4B
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B*/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --show_labels --only_stat_table --tablefmt latex


-----

# Low dim projection experiments tab:low_dim_projection_results
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_32_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_256_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_512_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_32_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_64_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_128_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_256_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_512_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_32_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_64_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_128_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_256_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_512_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_32_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_64_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_128_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_256_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_512_lowproj/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --midrule_indicies 3 9 15 \
  --show_labels --only_stat_table --tablefmt latex


# sl_4096_Llama-3.2-1B_lowdim_32_lowproj                                  153.20±78.77    1472.99±975.35         3278.20±1525.41   18.90±8.70               74                nan          nan  552.0315±344.6267                            nan
# sl_4096_Llama-3.2-1B_lowdim_64_lowproj                                  202.20±54.68    2622.75±1646.04        4250.30±1677.97   24.30±3.77               86                nan          nan  798.5700±130.6292                            nan
# sl_4096_Llama-3.2-1B_lowdim_128_lowproj                                 218.10±65.12    4322.61±2204.61        6704.80±4447.70   22.50±3.85               79                nan          nan  714.3459±353.7436                            nan
# sl_4096_Llama-3.2-1B_lowdim_256_lowproj                                 197.80±47.53    2801.80±624.13         3466.60±875.16    19.30±5.85              102                nan          nan  684.6490±352.5690                            nan
# sl_4096_Llama-3.2-1B_lowdim_512_lowproj                                 198.20±93.13    4528.04±2197.06        3720.20±2132.93   18.00±4.27               88                nan          nan  583.8316±545.1842                            nan

# sl_4096_Llama-3.2-3B_lowdim_32_lowproj                                  907.70±233.74   5467.89±2589.14        0.00±0.00         30.10±4.91              137                nan          nan  3103.6157±677.5746                           nan
# sl_4096_Llama-3.2-3B_lowdim_64_lowproj                                  887.50±261.42   5843.63±2653.04        0.00±0.00         31.30±5.33              141                nan          nan  2849.2596±923.0871                           nan
# sl_4096_Llama-3.2-3B_lowdim_128_lowproj                                 773.60±292.93   5845.21±3545.74        0.00±0.00         26.60±5.52              122                nan          nan  2518.2422±1182.5698                          nan
# sl_4096_Llama-3.2-3B_lowdim_256_lowproj                                 738.80±317.66   10639.99±6516.14       0.00±0.00         28.10±9.32              129                nan          nan  2575.4518±1240.2663                          nan
# sl_4096_Llama-3.2-3B_lowdim_512_lowproj                                 811.00±215.78   16793.89±7327.45       0.00±0.00         28.70±7.18              136                nan          nan  2796.1188±761.0452                           nan

# sl_4096_Meta-Llama-3.1-8B_lowdim_32_lowproj                                  1745.00±305.99  10250.32±3256.38   0.00±0.00      35.70±5.10              140                nan          nan  5312.0572±329.6113                           nan
# sl_4096_Meta-Llama-3.1-8B_lowdim_256_lowproj                                 1730.80±384.35  21447.77±5582.75   0.00±0.00      29.40±3.29              147                nan          nan  4869.9974±1679.8964                          nan
# sl_4096_Meta-Llama-3.1-8B_lowdim_512_lowproj                                 1652.90±404.48  25851.58±7527.98   0.00±0.00      36.20±5.88              162                nan          nan  5071.8260±798.6758                           nan

# ---

# sl_4096_pythia-1.4b_lowdim_32_lowproj                                  358.20±80.64    1550.61±953.66    8742.30±2395.37   15.10±3.01               82                nan          nan  1136.5011±241.3849                           nan
# sl_4096_pythia-1.4b_lowdim_64_lowproj                                  392.80±96.62    1919.65±1190.73   8196.50±1920.41   16.70±3.69               87                nan          nan  1240.0505±286.2224                           nan
# sl_4096_pythia-1.4b_lowdim_128_lowproj                                 373.00±81.81    1914.51±1133.27   9013.20±1934.87   15.80±3.22               89                nan          nan  1168.0420±240.0144                           nan
# sl_4096_pythia-1.4b_lowdim_256_lowproj                                 375.50±100.66   2134.77±1312.39   7590.50±2342.10   16.50±2.16               87                nan          nan  1188.9791±313.6299                           nan
# sl_4096_pythia-1.4b_lowdim_512_lowproj                                 415.50±76.30    2863.47±1432.90   7473.30±1632.56   18.00±2.49               98                nan          nan  1332.7256±199.7924                           nan

# sl_4096_pythia-410m_lowdim_32_lowproj                                  102.60±34.78    1226.35±573.66    10145.40±3758.45  13.40±3.41               73                nan          nan  351.2602±112.3417                            nan
# sl_4096_pythia-410m_lowdim_64_lowproj                                  102.80±30.96    2059.96±1136.54   8350.00±4104.56   15.50±3.85               84                nan          nan  352.2430±117.3033                            nan
# sl_4096_pythia-410m_lowdim_128_lowproj                                 133.80±33.00    3382.83±954.02    11265.30±2285.20  18.90±2.34               97                nan          nan  460.4902±72.9850                             nan
# sl_4096_pythia-410m_lowdim_256_lowproj                                 136.50±40.79    6811.96±1644.71   10353.20±3461.78  18.60±4.13              100                nan          nan  480.6429±122.4841                            nan
# sl_4096_pythia-410m_lowdim_512_lowproj                                 155.90±40.95    9513.60±4638.55   11936.50±3802.79  21.10±2.62               95                nan          nan  528.5584±110.4659                            nan

# sl_4096_pythia-160m_lowdim_32_lowproj                                  9.50±3.23       264.30±214.07     1876.60±707.52    2.50±1.86                25                nan          nan  20.6846±23.0741                              nan
# sl_4096_pythia-160m_lowdim_64_lowproj                                  12.70±6.26      595.78±447.19     2865.50±1497.25   3.60±3.38                35                nan          nan  50.1697±26.8477                              nan
# sl_4096_pythia-160m_lowdim_128_lowproj                                 10.40±4.34      742.06±638.37     2547.10±1830.70   2.90±2.39                25                nan          nan  36.6760±25.5980                              nan
# sl_4096_pythia-160m_lowdim_256_lowproj                                 10.70±2.76      1155.91±587.25    2486.50±978.28    2.30±1.49                26                nan          nan  36.9150±20.4974                              nan
# sl_4096_pythia-160m_lowdim_512_lowproj                                 15.50±6.59      3490.27±2241.99   3259.30±1451.67   5.00±3.29                40                nan          nan  64.5877±41.3813                              nan

# ---
# | sl_4096_SmolLM2-1.7B_lowdim_32_lowproj    | 335.80±78.70   | 543.23±217.84    | 0.00±0.00       | 14.20±1.99   |            64 |               nan |         nan | 1006.9515±284.3068    |                       nan |
# | sl_4096_SmolLM2-1.7B_lowdim_64_lowproj    | 403.20±77.73   | 777.87±167.10    | 0.00±0.00       | 13.80±1.94   |            68 |               nan |         nan | 1195.0204±176.9075    |                       nan |
# | sl_4096_SmolLM2-1.7B_lowdim_128_lowproj   | 431.90±115.59  | 1236.34±653.01   | 0.00±0.00       | 13.90±2.51   |            62 |               nan |         nan | 1252.1781±287.6511    |                       nan |
# | sl_4096_SmolLM2-1.7B_lowdim_256_lowproj   | 487.70±152.65  | 2264.61±1404.78  | 0.00±0.00       | 16.00±4.47   |            63 |               nan |         nan | 1448.5695±402.2275    |                       nan |
# | sl_4096_SmolLM2-1.7B_lowdim_512_lowproj   |


# | sl_4096_SmolLM2-360M_lowdim_32_lowproj  | 54.60±34.93    | 178.49±134.11    | 4265.80±1802.98 | 7.50±1.12    |            30 |               nan |         nan | 220.8946±110.5443     |                       nan |
# | sl_4096_SmolLM2-360M_lowdim_64_lowproj  | 49.60±16.10    | 210.17±170.73    | 4294.80±1328.27 | 7.10±1.87    |            33 |               nan |         nan | 215.5386±66.9511      |                       nan |
# | sl_4096_SmolLM2-360M_lowdim_128_lowproj | 54.80±23.66    | 297.34±185.53    | 4895.80±2744.93 | 6.60±1.20    |            32 |               nan |         nan | 231.1017±83.2736      |                       nan |
# | sl_4096_SmolLM2-360M_lowdim_256_lowproj | 59.10±24.41    | 598.59±775.82    | 5069.40±3173.59 | 7.10±1.64    |            23 |               nan |         nan | 248.8198±89.8057      |                       nan |
# | sl_4096_SmolLM2-360M_lowdim_512_lowproj |

# | sl_4096_SmolLM2-135M_lowdim_32_lowproj  | 30.00±17.80    | 66.01±44.06      | 0.00±0.00       | 5.90±2.70    |            36 |               nan |         nan | 126.0149±65.0823      |                       nan |
# | sl_4096_SmolLM2-135M_lowdim_64_lowproj  | 63.50±34.73    | 406.65±445.52    | 0.00±0.00       | 7.50±3.38    |            34 |               nan |         nan | 260.2847±129.7826     |                       nan |
# | sl_4096_SmolLM2-135M_lowdim_128_lowproj | 59.40±14.78    | 353.00±146.51    | 0.00±0.00       | 8.70±1.55    |            48 |               nan |         nan | 257.4038±69.7595      |                       nan |
# | sl_4096_SmolLM2-135M_lowdim_256_lowproj | 77.70±23.19    | 931.10±757.15    | 0.00±0.00       | 10.20±1.54   |            43 |               nan |         nan | 313.9395±82.2531      |                       nan |
# | sl_4096_SmolLM2-135M_lowdim_512_lowproj |


# Allignment
# TODO сделать график
# Full experiments list tab:full_activation_alignment_and_low_dim_projections
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_2/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_4/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_16/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_24/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_32/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_64_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_128_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_loss_cosine_hybrid_1.0_align_4/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_loss_cosine_hybrid_1.0_align_16/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_loss_cosine_hybrid_1.0_align_20/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_lowdim_64_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_lowdim_128_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Qwen3-4B_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_loss_cosine_hybrid_1.0_align_4/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_loss_cosine_hybrid_1.0_align_16/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_loss_cosine_hybrid_1.0_align_20/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_64_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_128_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_loss_cosine_hybrid_1.0_align_4/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_loss_cosine_hybrid_1.0_align_16/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_loss_cosine_hybrid_1.0_align_20/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_64_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_128_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --midrule_indicies 5 9 13 17 25 \
  --show_labels --only_stat_table --tablefmt latex



# tab:alignment_and_lowdim_projection
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_loss_cosine_hybrid_1.0_align_4/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --midrule_indicies 2 5 8 \
  --show_labels --only_stat_table --tablefmt latex


# tab:all_progressive_modifications
# TODO add some progressive + lowdim gemma experiment artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8
PYTHONPATH=./src:. python scripts/paper/low_dimesional.py \
  --checkpoints \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_32_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lowdim_32_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lr_0.5/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_256_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_pythia-1.4b_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_loss_cosine_hybrid_1.0_align_4/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_256_lowproj/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_SmolLM2-1.7B_lowdim_256_lowproj_loss_cosine_hybrid_1.0_align_8/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lr_0.1/progressive_prefixes \
    artifacts/experiments_progressive/sl_4096_gemma-3-4b-pt_lowdim_256_lowproj/progressive_prefixes \
  --n_components 4 \
  --sample_id 0 \
  --midrule_indicies 3 7 11 \
  --show_labels --only_stat_table --tablefmt latex


