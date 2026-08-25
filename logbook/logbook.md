# Logbook

## Meeting 1: 26th May

Notes:

- Build a fault map for Bristol Channel Fault zone
- Concentrate on developing a model after the field trip.
- Look into geological map of Lizard and BCBF area for possible migration pathways.

Progress before next meeting:

- Worked on fault map for Bristol Channel Fault Zone
- Literature review for Variscan Suture Front (links BCFZ and Bray Fault in Paris Basin)
- set-up repository structure.
- developed substantial parts of the code-base foundation
- competed deep learning to identify small subcircular depressions in southwest england
- completed geostatistics on clusters of SCDs in three locations, Mendip Hills, Weymouth and Taunton.
- developed an outline for field investigations,and identified the WCHF fault as a major target.

## Meeting 2: 11th June

Aim: 

- Pre-field trip meeting to discuss logistics, field target selection and specific objectives of fieldwork.
- Discuss my desk study findings in this regard.

## Field Trip 

- 7 Day field trip.
- Considerable discussion on project direction informed by soil gas campaign results.

## Direction Notes:

- Experimenting with UNET/CentreNet in colab, using dataset created from DataStack. At inference, the model will need to see both 50m SCDs and km-scale SCDs, this scale problem is not solvable with an FPN or aggressive ASPP.
- Instead, I've read a few papers on SNIP (Scale Normalised Image Pyramids), and think this is a really good way to learn object detection irrespective of scale. At inference, a Image pyramid is used. This will be slow, but accuracy will be unrivalled from other methods from what I've seen so far in my experimentation.


## Direction Notes (11/07)

- Looking into medical imaging techniques (Mask RCNN).
- Aiming at transfer learning to limit overfitting problem on small dataset.
- Will try to put a CentreNet head onto Mask RCNN backbone.
- Presented research to H2 research group and discussed next steps for general hydrogen research in the department.

 
## Meeting Notes

- Image Pyramid idea is novel, implementation needs to be improved and cross-geography LOOCV at deployment level.

## Direction Notes (10/08)

- Finalising MaskRCNN implementation and running final training and deployment runs.
- Figure drafts compiled and writing plan outlined.
- Image Pyramid parameters finalised
- Deployment runs completed on folded-out regions

## Direction Notes (20/8)

- Paper draft finished and repo structured. Working on final revisions to the paper based on supervisor feedback.
- Primarily aiming to cut words and refactor for conciseness as a priority, hoping for around 4000 words at the end.

## Meeting Notes (21/08)

Meeting with Valentin. 

- Discussed draft feedback
- Discussed what parts of the paper will be changed post-IRP when we submit for publication.
- Discussed possible incorporation of some of these initial changes into the final IRP report for completeness.
- Overall, very happy with the work and looking forward to developing further in October.

## Progress Notes (25/08)

- Incorporated details of Cornwall field trip into results and discussion.
- Added clarity on background measurements (Table 2 in report)
- Increased quality and coverage of documentation throughout repository and added report assets, raw data, etc.

