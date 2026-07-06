# Angular Chunk Selection Bias Check

This report checks whether later-chunk angular AUROC is affected by only longer responses reaching later chunks.

## Question Set Differences

| chunk | has chunk | questions | mean acc | median acc | mean len | median len |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | False | 83 | 0.6526 | 0.7500 | 265.9 | 275.4 |
| 2 | True | 1034 | 0.4479 | 0.4444 | 888.3 | 952.7 |
| 3 | False | 102 | 0.6654 | 0.7735 | 293.3 | 298.6 |
| 3 | True | 1015 | 0.4428 | 0.4286 | 897.2 | 955.7 |

## Native vs Common-Question AUROC

Common-question AUROC restricts both chunk 2 and chunk 3 to the same question set.

| layer | chunk | feature | native q | native AUROC+ | common q | common AUROC+ | common AUROC- |
|---:|---:|---|---:|---:|---:|---:|---:|
| 24 | 3 | cos_min | 782 | 0.5853 | 782 | 0.5853 | 0.4147 |
| 36 | 3 | cos_mean | 782 | 0.5798 | 782 | 0.5798 | 0.4202 |
| 24 | 3 | cos_p10 | 782 | 0.5636 | 782 | 0.5636 | 0.4364 |
| 36 | 2 | cos_mean | 894 | 0.5635 | 893 | 0.5634 | 0.4366 |
| 36 | 3 | cos_min | 782 | 0.5621 | 782 | 0.5621 | 0.4379 |
| 24 | 2 | cos_min | 894 | 0.5563 | 893 | 0.5563 | 0.4437 |
| 24 | 3 | cos_mean | 782 | 0.5557 | 782 | 0.5557 | 0.4443 |
| 24 | 3 | av_cos_mean | 782 | 0.5514 | 782 | 0.5514 | 0.4486 |
| 24 | 2 | cos_p10 | 894 | 0.5497 | 893 | 0.5500 | 0.4500 |
| 24 | 2 | av_cos_mean | 894 | 0.5460 | 893 | 0.5461 | 0.4539 |
| 24 | 2 | cos_mean | 894 | 0.5425 | 893 | 0.5427 | 0.4573 |
| 36 | 2 | av_cos_mean | 894 | 0.5420 | 893 | 0.5423 | 0.4577 |
| 36 | 3 | av_cos_mean | 782 | 0.5420 | 782 | 0.5420 | 0.4580 |
| 36 | 2 | cos_min | 894 | 0.5409 | 893 | 0.5412 | 0.4588 |
| 36 | 3 | cos_p10 | 782 | 0.5357 | 782 | 0.5357 | 0.4643 |
| 36 | 2 | cos_p10 | 894 | 0.5281 | 893 | 0.5284 | 0.4716 |
| 24 | 3 | spike_rate_90 | 782 | 0.5001 | 782 | 0.5001 | 0.4999 |
| 24 | 2 | spike_rate_90 | 894 | 0.4977 | 893 | 0.4977 | 0.5023 |
| 36 | 2 | spike_rate_90 | 894 | 0.4397 | 893 | 0.4399 | 0.5601 |
| 36 | 3 | spike_rate_90 | 782 | 0.4351 | 782 | 0.4351 | 0.5649 |
