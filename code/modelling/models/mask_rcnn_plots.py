import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_preds(
    model,
    val_loader,
    device,
    score_threshold=0.5,
    mask_threshold=0.5,
    max_images=6,
):
    model.eval()

    plotted = 0

    with torch.no_grad():
        for images, targets in val_loader:

            # Mask R-CNN expects a list of [C, H, W] tensors
            images_device = [img.to(device) for img in images]

            # Standard torchvision output:
            # list of dictionaries containing boxes, labels, scores and masks
            outputs = model(images_device)

            for i, output in enumerate(outputs):

                image = images[i].detach().cpu()

                # Display DEM for reference / interpretaton
                img_display = image[0].numpy()

                # gt at instance level (different to sematnci segmantetaion)
                gt_instance_masks = targets[i]["masks"].detach().cpu()

                if gt_instance_masks.shape[0] > 0:
                    gt_mask = gt_instance_masks.bool().any(dim=0).numpy()
                else:
                    gt_mask = np.zeros(
                        image.shape[-2:],
                        dtype=bool,
                    )

                # preds
                scores = output["scores"].detach().cpu()
                pred_instance_masks = output["masks"].detach().cpu()

                keep = scores >= score_threshold
                pred_instance_masks = pred_instance_masks[keep]

                if pred_instance_masks.shape[0] > 0:
                    pred_mask = (
                        pred_instance_masks[:, 0] >= mask_threshold
                    ).any(dim=0).numpy()
                else:
                    pred_mask = np.zeros(
                        image.shape[-2:],
                        dtype=bool,
                    )

                # plot
                fig, axes = plt.subplots(1, 3, figsize=(10, 4))

                axes[0].imshow(img_display, cmap="terrain")
                axes[0].set_title("Input")
                axes[0].axis("off")

                axes[1].imshow(img_display, cmap="gray")
                axes[1].imshow(
                    np.ma.masked_where(~gt_mask, gt_mask),
                    alpha=0.5,
                    cmap="Greens",
                )
                axes[1].set_title("Ground truth")
                axes[1].axis("off")

                axes[2].imshow(img_display, cmap="gray")
                axes[2].imshow(
                    np.ma.masked_where(~pred_mask, pred_mask),
                    alpha=0.5,
                    cmap="Reds",
                )
                axes[2].set_title(
                    f"Prediction\nscore ≥ {score_threshold}"
                )
                axes[2].axis("off")

                plt.tight_layout()
                plt.show()

                plotted += 1

                if plotted >= max_images:
                    return