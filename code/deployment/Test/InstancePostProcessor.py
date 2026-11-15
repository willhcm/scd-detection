from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

from deployment.PostProcesser import PostProcessor


@dataclass
class InstanceRegion:
    coords: np.ndarray
    centroid: tuple
    equivalent_diameter_area: float
    mean_intensity: float


class InstancePostProcessor(PostProcessor):
    """Apply the existing veto feature construction to individual instances."""

    @staticmethod
    def _region(instance):
        coords = np.asarray(instance["coords"], dtype=np.intp)
        area = max(len(coords), 1)
        centroid = tuple(coords.mean(axis=0))
        return InstanceRegion(
            coords=coords,
            centroid=centroid,
            equivalent_diameter_area=float(np.sqrt(4.0 * area / np.pi)),
            mean_intensity=float(instance["score"]),
        )

    def predict_instances(self, instances, transform, crs):
        accepted, rejected, probabilities = [], [], []
        batch_dem, batch_candidate, batch_scalars, batch_instances = [], [], [], []

        def flush_batch():
            if not batch_instances:
                return

            dem_tensor = torch.from_numpy(np.stack(batch_dem)).to(
                self.device, dtype=torch.float32
            )
            candidate_tensor = torch.from_numpy(np.stack(batch_candidate)).to(
                self.device, dtype=torch.float32
            )
            scalar_tensor = torch.from_numpy(np.stack(batch_scalars)).to(
                self.device, dtype=torch.float32
            )
            with torch.inference_mode():
                logits = self.model(
                    dem_tensor,
                    candidate_tensor,
                    scalar_tensor,
                ).view(-1)
                batch_probabilities = torch.sigmoid(logits).cpu().numpy()

            for instance, probability in zip(batch_instances, batch_probabilities):
                probability = float(probability)
                probabilities.append(probability)
                if probability >= self.veto_threshold:
                    accepted.append(instance)
                else:
                    rejected.append(instance)

            batch_dem.clear()
            batch_candidate.clear()
            batch_scalars.clear()
            batch_instances.clear()

        progress = tqdm(instances, desc="Vetoing instances", unit="instance")
        for instance in progress:
            region = self._region(instance)
            tile = self._make_tile(region, transform, crs)
            if tile is None:
                rejected.append(instance)
                continue

            batch_dem.append(tile["image"])
            batch_candidate.append(tile["candidate"])
            batch_scalars.append(tile["scalar_features"])
            batch_instances.append(instance)
            if len(batch_instances) >= self.batch_size:
                flush_batch()

        flush_batch()
        progress.close()
        print(
            f"{len(accepted)} instances retained, "
            f"{len(rejected)} instances rejected."
        )
        return accepted, rejected, probabilities
