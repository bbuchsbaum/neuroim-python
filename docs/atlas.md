# Atlas Module

`neuroim.atlas` is the thin in-core atlas layer. It gives atlas images a typed
identity, source policy, label metadata, spatial contract, and conversion into
the clustered containers used by `NeuroVec.parcel_means`.

The module does not bundle large atlas data, import TemplateFlow at package
load, or own interactive visualization. Broad catalogs, live fetchers, surface
rendering, and heavyweight data backends belong in optional extras or a
separate package.

## Source Policy

Schaefer is the first supported source policy because its upstream story is
clean:

- canonical source: ThomasYeoLab/CBIG Schaefer2018
- citation DOI: `10.1093/cercor/bhx179`
- preferred delivery backend: TemplateFlow when available
- fallback delivery backend: pinned CBIG artifact
- Nilearn role: parity/reference only, not the data source

The current core constructor accepts an already-loaded label image:

```python
import neuroim as ni
from neuroim.atlas import AtlasLabel, schaefer_200

label_img = ni.read_image("Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz", type="vol")
atlas = schaefer_200(
    label_img,
    labels=[
        AtlasLabel(1, "LH_Default_A", hemi="left", network="Default"),
        AtlasLabel(2, "RH_Visual_A", hemi="right", network="Visual"),
    ],
    delivery_backend="templateflow",
)

parcel_ts = bold.parcel_means(atlas)
print(parcel_ts.atlas_provenance.citation_doi)
```

The first implementation deliberately builds on the Scenario 11 parcel-time
series path: `bold.parcel_means(atlas)` projects a typed atlas into the same
clustered reduction machinery that already proved numeric parity against a raw
`nibabel`+`numpy` baseline. The atlas object adds the missing source layer:
which artifact supplied the labels, what label table was used, what image/table
hashes identify it, and how much confidence neuroim should attach to that
source.

## Provider Contracts

Core also defines dependency-free extension contracts for future providers:

- `AtlasSpec`: a request object, for example Schaefer 200 parcels, 7 networks,
  volume representation, MNI space, 2 mm resolution.
- `AtlasArtifact`: one source artifact used by a provider, such as a NIfTI label
  image, label table, transform, or surface annotation.
- `AtlasProvider`: a structural protocol with `get(spec) -> VolumetricAtlas`.

These contracts are intentionally not implementations. A future TemplateFlow,
CBIG, local-file, or neuromaps provider can satisfy them without `neuroim`
depending on that package. Importing `neuroim` or `neuroim.atlas` must stay
free of TemplateFlow, neuromaps, nilearn, downloader, datalad, or catalog
imports.

## Glasser/HCP-MMP1.0 Audit

Glasser is not exposed as a core fetcher yet. The R `neuroatlas` package uses
several useful but differently trustworthy sources:

| Representation | Source | Current decision |
|---|---|---|
| volume | PennLINC/xcpEngine `glasser360MNI.nii.gz` plus `glasser360NodeNames.txt` | Runtime-stable, but template identity is not explicit. A Python wrapper may expose this only with `template_space="MNI152_unspecified"` or equivalent and `confidence="uncertain"`. |
| volume | Raj-Lab-UCSF `Human_Brain_Atlases-glasser` `MMP_in_MNI_corr.nii.gz` | Cleaner MNI2009c provenance, but mirrors may serve a git-annex pointer stub. A Python wrapper must verify the downloaded artifact is a real NIfTI payload and record a content hash before claiming high confidence. |
| labels | xcpEngine `glasser360NodeNames.txt` | Usable for stable naming, but if paired with a non-xcpEngine volume, provenance must say so explicitly. |
| surface | Kathryn Mills Figshare `lh.HCP-MMP1.annot` / `rh.HCP-MMP1.annot`, DOI `10.6084/m9.figshare.3498446`, with TemplateFlow fsaverage geometry | Good source story, but out-of-core until neuroim has a mature surface container story. |

TemplateFlow geometry is appropriate for the surface path; it is not, by
itself, a solved volumetric Glasser source policy. The Python rule is therefore
conservative: do not ship a `neuroim.atlas.glasser_*` core constructor until
the selected volume backend is hash-verified and its confidence can be stated
without laundering source ambiguity.
