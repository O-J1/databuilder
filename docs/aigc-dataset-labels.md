# AIGC dataset label audit

Audit for [`examples/aigc-datasets.toml`](../examples/aigc-datasets.toml). “Real” means a human/camera-source
image and “fake” means an AI-generated or AI-manipulated image. 

It does not turn URL fields into per-image HTTP requests. URL-only, metadata-only, or
provenance-ambiguous repositories therefore use `download_only = true`: their
snapshot is retained but they cannot silently enter the labelled manifest.

| Dataset | Labels admitted to manifest | Generator/provenance rule | Handling note |
|---|---|---|---|
| [terminusresearch/nijijourney-v6-520k-raw](https://huggingface.co/datasets/terminusresearch/nijijourney-v6-520k-raw) | fake only | `nijijourney-v6` | Generated-image tar archives. |
| [terminusresearch/midjourney-v6-520k-raw](https://huggingface.co/datasets/terminusresearch/midjourney-v6-520k-raw) | fake only | `midjourney-v6` | Generated-image tar archives. |
| [bitmind/GenImage_MidJourney](https://huggingface.co/datasets/bitmind/GenImage_MidJourney) | fake only | `midjourney` | MidJourney subset of GenImage. |
| [Ashenone3/Midjourney-23M](https://huggingface.co/datasets/Ashenone3/Midjourney-23M) | fake only | `midjourney` | WebDataset image tars. |
| [Photoroom/midjourney-v6-recap](https://huggingface.co/datasets/Photoroom/midjourney-v6-recap) | fake only | `midjourney-v6` | Generated-image parquet. |
| [SaunakSS/nano-banana-synthetic-images-1500](https://huggingface.co/datasets/SaunakSS/nano-banana-synthetic-images-1500) | fake only | `nano-banana` | Card identifies the corpus as 100% synthetic. |
| [FlameF0X/nano-banana-pro-gen-zh-en](https://huggingface.co/datasets/FlameF0X/nano-banana-pro-gen-zh-en) | fake only | `nano-banana-pro` | Embedded images in Arrow files. |
| [julienlucas/midjourney-dalle-sd-nanobananapro-dataset](https://huggingface.co/datasets/julienlucas/midjourney-dalle-sd-nanobananapro-dataset) | real and fake | static mixed-generator group | Physical class `0` is fake and `1` is real; the config reverses databuilder’s usual numeric-name convention explicitly. |
| [bitmind/nano-banana](https://huggingface.co/datasets/bitmind/nano-banana) | fake only | `nano-banana` | Generated-image parquet. |
| [Dream000/gpt-image](https://huggingface.co/datasets/Dream000/gpt-image) | fake only | `gpt-image` | Generated-image zip. |
| [Tungtom2004/Output_dataset_gptImage1](https://huggingface.co/datasets/Tungtom2004/Output_dataset_gptImage1) | fake only | `gpt-image-1` | Direct generated image files. |
| [mhussainahmad/sdxl-1024-100k](https://huggingface.co/datasets/mhussainahmad/sdxl-1024-100k) | fake only | `sdxl` | SDXL-generated parquet images. |
| [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) | real and fake | `model` column, excluding reported corrupt groups | Card/schema provides `real`/`fake` labels and the source model. `flux.1-dev`, `sd-3.5`, `sdxl-realvis-v5`, and `sd-1.5-dreamshaper` are excluded because of the [maintainer-acknowledged image/prompt mismatch](https://huggingface.co/datasets/ComplexDataLab/OpenFake/discussions/2); SANA is excluded because its rows are reported as invalid noise images. |
| [Rapidata/Ideogram-V2_t2i_human_preference](https://huggingface.co/datasets/Rapidata/Ideogram-V2_t2i_human_preference) | fake only | `model1`/`model2` columns | Both members of each preference pair are generated; both image fields are emitted. |
| [Rapidata/Seedream-3_t2i_human_preference](https://huggingface.co/datasets/Rapidata/Seedream-3_t2i_human_preference) | fake only | `model1`/`model2` columns | Both pair members are generated. |
| [Rapidata/Flux-2-pro_t2i_human_preference](https://huggingface.co/datasets/Rapidata/Flux-2-pro_t2i_human_preference) | fake only | `model1`/`model2` columns | Both pair members are generated. |
| [Junwei-Xi/EvalGEN](https://huggingface.co/datasets/Junwei-Xi/EvalGEN) | fake only | archive/folder generator name | Evaluation classes name image generators, not real/fake classes. |
| [data-is-better-together/open-image-preferences-v1](https://huggingface.co/datasets/data-is-better-together/open-image-preferences-v1) | fake only | FLUX.1-dev or SD 3.5 Large per field | All four configured image fields are model outputs. |
| [ash12321/seedream-4.5-generated-2k](https://huggingface.co/datasets/ash12321/seedream-4.5-generated-2k) | fake only | `seedream-4.5` | Direct generated images. |
| [Skywork/unipic_seedream_4images](https://huggingface.co/datasets/Skywork/unipic_seedream_4images) | fake only | `seedream` | Rows contain real input references plus a generated output; only `output_image` is extracted from the split zip. |
| [Skywork/unipic_seedream_5images](https://huggingface.co/datasets/Skywork/unipic_seedream_5images) | fake only | `seedream` | Only `output_image` is extracted. |
| [Skywork/unipic_seedream_6images](https://huggingface.co/datasets/Skywork/unipic_seedream_6images) | fake only | `seedream` | Only `output_image` is extracted. |
| [Skywork/unipic_nano_2images](https://huggingface.co/datasets/Skywork/unipic_nano_2images) | fake only | `nano-banana` | Only the generated `output_image`, not input references, is extracted. |
| [Skywork/unipic_nano_3images](https://huggingface.co/datasets/Skywork/unipic_nano_3images) | fake only | `nano-banana` | Only the generated `output_image` is extracted. |
| [terminusresearch/ideogram-75k](https://huggingface.co/datasets/terminusresearch/ideogram-75k) | fake only | `ideogram` | Generated images stored as numbered chunks of one tar stream; `multipart_tar` reads across chunk boundaries without creating a combined temporary archive. |
| [deepthink8/kling-ai-images](https://huggingface.co/datasets/deepthink8/kling-ai-images) | fake only | `kling` | The config selects archives under the fake image tree. |
| [nyuuzyou/klingai](https://huggingface.co/datasets/nyuuzyou/klingai) | fake only | `kling` | Only `resource_*.zip` image payloads are selected; cover and video metadata are excluded. |
| [hastylol/nai3](https://huggingface.co/datasets/hastylol/nai3) | fake only | `novelai-v3` | NAI v3 generated-image zips. |
| [Yejy53/Nano-consistent-150k](https://huggingface.co/datasets/Yejy53/Nano-consistent-150k) | fake only | `nano-banana` | The card is empty; this is an explicit, provisional inference from the repository name/layout. |
| [JamalLee/pre-2026](https://huggingface.co/datasets/JamalLee/pre-2026) | real and fake | `generator` column | The alias redirects to `JamalLee/Omni-Fake-SET`. Only its image parquet subset is selected; `real` stays real while `full_synthetic` and `tampered` map to fake. |
| [lingcco/FakeClue](https://huggingface.co/datasets/lingcco/FakeClue) | fake only | `fakeclue` | Train/test archives contain the generated benchmark images. |
| [kafked/anycrap](https://huggingface.co/datasets/kafked/anycrap) | none in the main config (raw snapshot) | generated output metadata | `image_url` is external. The standalone `scripts/download_url_datasets.py` selects rows with `has_real_image = true`; downloaded outputs remain fake-labelled. |
| [lehduong/seaart-hq](https://huggingface.co/datasets/lehduong/seaart-hq) | none in the main config (raw snapshot) | `seaart` output metadata | Parquet rows contain external `url` values. The standalone URL downloader can fetch them without changing the main config. |
| [dsixteen/Niji_1_11](https://huggingface.co/datasets/dsixteen/Niji_1_11) | fake only | `niji` | Generated-image parquet. |
| [EricY05/lvlm-ood-fake-data](https://huggingface.co/datasets/EricY05/lvlm-ood-fake-data) | real and fake | static `lvlm-ood` group | `id_real`/`ood_real` map to real; `fake_id`/`fake_ood` map to fake. |
| [ThaneJoss/SDAIE](https://huggingface.co/datasets/ThaneJoss/SDAIE) | real and fake | subset generator columns/static source | Four disjoint config entries avoid duplicate payloads: `aigi_test` and `cnnspot_trainset` use their label/generator columns; `exif_pretrain` and `photographic_10k` are real-only. |
| [shimei123/Genimage](https://huggingface.co/datasets/shimei123/Genimage) | real and fake | one static generator per archive | The repository has no card body. Its eight archives mirror the official GenImage layout: `ai` maps to fake and `nature` maps to real. Disjoint config entries retain ADM, BigGAN, Midjourney, SD 1.4/1.5, VQDM, GLIDE, and Wukong provenance. |
| [saberzl/So-Fake-Set](https://huggingface.co/datasets/saberzl/So-Fake-Set) | real and fake | `generator` column; null becomes `unknown` for real rows | The card defines `real` as real; `full_synthetic` and `tampered` both map to fake. Only the primary `image` field is materialized, not the manipulation mask. |
| [alecrespi/RRDataset-CV-Project2](https://huggingface.co/datasets/alecrespi/RRDataset-CV-Project2) | real and fake | `rrdataset-original` or `redigital` by archive | Viewer row keys identify `ai` and `real` label folders. The original train/validation archive and held-out redigital test archive are disjoint entries; the latter is forced to `test`. |
| [saberzl/So-Fake-OOD](https://huggingface.co/datasets/saberzl/So-Fake-OOD) | real and fake; test only | `generator` column; null becomes `unknown` for real rows | Card ClassLabels `0`/`1`/`2` mean real/tampered/full-synthetic. Tampered and full-synthetic map to fake, and every row is forced to `test`. Only `image`, not `mask`, is materialized. |
