Datasets:

- Beauty:
    - Filter 5-core and filter items without both text and vision.
    - Statistic:

| items  | users  | interactions |
|--------|--------|--------------|
| 12 101 | 22 363 | 198 502      |

    - File's details:

| File   | Details                                 | Link | Note                   |
|--------|-----------------------------------------|------|------------------------|
| tst    | seqs with full interaction for testing  |      |                        |
| seq    | seqs with masked last item for training |      |                        |
| trn    | generated from seq                      |      |                        |
| neg    | sample 99 neg items for each seq        |      |                        |
| text   | text feature                            |      | full                   |
| vision | vision feature                          |      | 7 items missing vision |
