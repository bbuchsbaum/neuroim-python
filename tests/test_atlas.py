import numpy as np
import pytest

import neuroim as ni
from neuroim.atlas import (
    AtlasArtifact,
    AtlasLabel,
    AtlasProvider,
    AtlasSpec,
    VolumetricAtlas,
    schaefer_200,
)


def _space(shape=(3, 3, 2)):
    return ni.NeuroSpace.from_affine(np.eye(4), shape)


def _bold_and_labels():
    space3 = _space()
    shape3 = tuple(int(d) for d in space3.dim)
    data = np.zeros(shape3 + (4,), dtype=float)
    labels = np.zeros(shape3, dtype=np.int32)

    labels[0, :, :] = 1
    labels[1:, :, :] = 2
    for t in range(data.shape[3]):
        data[..., t] = labels * 10 + t

    bold = ni.DenseNeuroVec(data, ni.NeuroSpace.from_affine(np.eye(4), data.shape))
    label_vol = ni.DenseNeuroVol(labels, space3)
    return bold, label_vol


def test_schaefer_constructor_records_source_policy():
    _, label_vol = _bold_and_labels()
    atlas = schaefer_200(
        label_vol,
        labels=(
            AtlasLabel(1, "LH_Default_A", hemi="left", network="Default"),
            AtlasLabel(2, "RH_Visual_A", hemi="right", network="Visual"),
        ),
        delivery_backend="templateflow",
    )

    assert isinstance(atlas, VolumetricAtlas)
    assert atlas.id == "schaefer-200-7"
    assert atlas.provenance.canonical_source == "ThomasYeoLab/CBIG"
    assert atlas.provenance.delivery_backend == "templateflow"
    assert atlas.provenance.citation_doi == "10.1093/cercor/bhx179"
    assert atlas.provenance.template_space == "MNI152NLin6Asym"
    assert atlas.provenance.image_hash != "none"
    assert atlas.provenance.label_table_hash != "none"
    assert atlas.label_names == ("LH_Default_A", "RH_Visual_A")


def test_typed_atlas_converts_to_clustered_volume_with_label_map():
    _, label_vol = _bold_and_labels()
    atlas = schaefer_200(
        label_vol,
        labels=(AtlasLabel(1, "parcel_one"), AtlasLabel(2, "parcel_two")),
    )

    cvol = atlas.to_clustered_vol()

    assert isinstance(cvol, ni.ClusteredNeuroVol)
    assert cvol.label_map == {"parcel_one": 1, "parcel_two": 2}
    assert cvol.atlas_provenance == atlas.provenance
    assert cvol.num_clusters() == 2


def test_parcel_means_accepts_typed_atlas_and_preserves_atlas_provenance():
    bold, label_vol = _bold_and_labels()
    atlas = schaefer_200(
        label_vol,
        labels=(AtlasLabel(1, "parcel_one"), AtlasLabel(2, "parcel_two")),
    )

    out = bold.parcel_means(atlas)

    assert isinstance(out, ni.ClusteredNeuroVec)
    assert out.atlas_provenance == atlas.provenance
    assert out.provenance.method_name == "parcel_means"
    assert out.provenance.n_voxels == 2
    np.testing.assert_allclose(out.ts[:, 0], np.array([10, 11, 12, 13]))
    np.testing.assert_allclose(out.ts[:, 1], np.array([20, 21, 22, 23]))


def test_parcel_means_distinguishes_atlases_with_different_label_tables():
    bold, label_vol = _bold_and_labels()
    atlas_a = schaefer_200(
        label_vol,
        labels=(AtlasLabel(1, "parcel_one"), AtlasLabel(2, "parcel_two")),
    )
    atlas_b = schaefer_200(
        label_vol,
        labels=(AtlasLabel(1, "parcel_one"), AtlasLabel(2, "renamed_two")),
    )

    out_a = bold.parcel_means(atlas_a)
    out_b = bold.parcel_means(atlas_b)

    assert out_a.atlas_provenance.image_hash == out_b.atlas_provenance.image_hash
    assert (
        out_a.atlas_provenance.label_table_hash
        != out_b.atlas_provenance.label_table_hash
    )


def test_parcel_means_rejects_wrong_space_typed_atlas():
    bold, label_vol = _bold_and_labels()
    shifted_affine = label_vol.space.affine.copy()
    shifted_affine[0, 3] = 10.0
    wrong_space = ni.NeuroSpace.from_affine(shifted_affine, label_vol.shape)
    atlas = schaefer_200(ni.DenseNeuroVol(label_vol.data, wrong_space))

    with pytest.raises(ValueError, match="spatial contract mismatch"):
        bold.parcel_means(atlas)


def test_atlas_is_lazy_submodule_not_root_public_budget():
    assert "atlas" not in ni.__all__
    assert ni.atlas.VolumetricAtlas is VolumetricAtlas
    assert ni.atlas.__all__ == [
        "AtlasLabel",
        "AtlasProvenance",
        "AtlasSpec",
        "AtlasArtifact",
        "AtlasProvider",
        "VolumetricAtlas",
        "schaefer_200",
    ]


def test_atlas_spec_and_artifact_support_future_providers():
    spec = AtlasSpec(
        family="schaefer",
        model="Schaefer2018",
        parcels=200,
        networks=7,
        space="MNI152NLin6Asym",
        resolution="2mm",
        backend="templateflow",
    )
    artifact = AtlasArtifact(
        role="label_image",
        source_url="https://www.templateflow.org",
        source_ref="tpl-MNI152NLin6Asym_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.nii.gz",
        hash="abc123",
        backend="templateflow",
        format="nii.gz",
        template_space=spec.space,
        confidence="high",
    )

    assert spec.family == "schaefer"
    assert spec.parcels == 200
    assert artifact.role == "label_image"
    assert artifact.template_space == "MNI152NLin6Asym"
    with pytest.raises(Exception):
        spec.parcels = 400
    with pytest.raises(Exception):
        artifact.hash = "changed"


def test_fake_provider_protocol_returns_volumetric_atlas():
    class FakeProvider:
        def get(self, spec: AtlasSpec) -> VolumetricAtlas:
            _, label_vol = _bold_and_labels()
            return schaefer_200(
                label_vol,
                networks=spec.networks or 7,
                labels=(AtlasLabel(1, "parcel_one"), AtlasLabel(2, "parcel_two")),
                delivery_backend=spec.backend or "fake",
            )

    provider = FakeProvider()
    spec = AtlasSpec(
        family="schaefer",
        parcels=200,
        networks=7,
        backend="fake",
    )
    atlas = provider.get(spec)

    assert isinstance(provider, AtlasProvider)
    assert isinstance(atlas, VolumetricAtlas)
    assert atlas.provenance.delivery_backend == "fake"
    assert atlas.provenance.image_hash != "none"


def test_volumetric_atlas_rejects_invalid_label_metadata():
    _, label_vol = _bold_and_labels()

    with pytest.raises(ValueError, match="unique"):
        schaefer_200(
            label_vol,
            labels=(AtlasLabel(1, "a"), AtlasLabel(1, "b")),
        )

    with pytest.raises(ValueError, match="background"):
        VolumetricAtlas(
            id="bad",
            name="bad",
            version="test",
            label_image=label_vol,
            labels=(AtlasLabel(0, "background"), AtlasLabel(1, "a")),
            background=0,
            provenance=schaefer_200(label_vol).provenance,
        )


def test_volumetric_atlas_rejects_invalid_label_images():
    _, label_vol = _bold_and_labels()

    with pytest.raises(TypeError, match="DenseNeuroVol"):
        schaefer_200(label_vol.as_logical())

    data4 = label_vol.data[..., np.newaxis]
    space4 = ni.NeuroSpace.from_affine(np.eye(4), data4.shape)
    with pytest.raises(ValueError, match="3D|3-D"):
        schaefer_200(ni.DenseNeuroVol(data4, space4))


def test_explicit_labels_must_match_non_background_image_ids():
    _, label_vol = _bold_and_labels()

    with pytest.raises(ValueError, match="missing metadata"):
        schaefer_200(label_vol, labels=(AtlasLabel(1, "parcel_one"),))

    with pytest.raises(ValueError, match="absent from label image"):
        schaefer_200(
            label_vol,
            labels=(
                AtlasLabel(1, "parcel_one"),
                AtlasLabel(2, "parcel_two"),
                AtlasLabel(3, "parcel_three"),
            ),
        )


def test_default_labels_infer_non_background_image_ids():
    _, label_vol = _bold_and_labels()

    atlas = schaefer_200(label_vol)

    assert tuple(atlas.label_ids.tolist()) == (1, 2)
    assert atlas.label_names == ("parcel_1", "parcel_2")


def test_nonzero_background_is_excluded_from_clusters():
    _, label_vol = _bold_and_labels()
    data = label_vol.data.copy()
    data[0, :, :] = 99
    custom = ni.DenseNeuroVol(data, label_vol.space)
    atlas = VolumetricAtlas(
        id="custom",
        name="custom",
        version="test",
        label_image=custom,
        labels=(AtlasLabel(2, "parcel_two"),),
        background=99,
        provenance=schaefer_200(label_vol).provenance,
    )

    cvol = atlas.to_clustered_vol()

    assert set(cvol.cluster_map) == {2}
    assert cvol.num_clusters() == 1


def test_atlas_hashes_are_stable_and_label_sensitive():
    _, label_vol = _bold_and_labels()
    labels_a = (AtlasLabel(1, "parcel_one"), AtlasLabel(2, "parcel_two"))
    labels_b = (AtlasLabel(1, "parcel_one"), AtlasLabel(2, "renamed_two"))

    atlas_a1 = schaefer_200(label_vol, labels=labels_a)
    atlas_a2 = schaefer_200(label_vol, labels=labels_a)
    atlas_b = schaefer_200(label_vol, labels=labels_b)

    assert atlas_a1.provenance.image_hash == atlas_a2.provenance.image_hash
    assert atlas_a1.provenance.label_table_hash == atlas_a2.provenance.label_table_hash
    assert atlas_a1.provenance.image_hash == atlas_b.provenance.image_hash
    assert atlas_a1.provenance.label_table_hash != atlas_b.provenance.label_table_hash
