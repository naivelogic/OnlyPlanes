import fiftyone as fo
import fiftyone.utils.coco as fouc

# And add model predictions
def fifty1_eval(dataset, eval_list, eval_view, labeltype):
    
    MAIN = eval_list[0][0]
    for x in eval_list:
        fouc.add_coco_labels(dataset,x[0],x[1],classes=None,label_type=labeltype)
    
    if eval_view:    
        # Evaluate the objects in the `predictions` field with respect to the
        #https://voxel51.com/docs/fiftyone/integrations/coco.html
        # objects in the `ground_truth` field
        results = dataset.evaluate_detections(MAIN,gt_field="ground_truth",method="coco",eval_key="eval")
        # Get the 10 most common classes in the dataset
        counts = dataset.count_values("ground_truth.detections.label")
        classes = sorted(counts, key=counts.get, reverse=True)

        # Print a classification report for the top-10 classes
        results.print_report(classes=classes)

        # Print some statistics about the total TP/FP/FN counts
        print("TP: %d" % dataset.sum("eval_tp"))
        print("FP: %d" % dataset.sum("eval_fp"))
        print("FN: %d" % dataset.sum("eval_fn"))

        # Create a view that has samples with the most false positives first, and
        # only includes false positive boxes in the `predictions` field
        from fiftyone import ViewField as F
        view = (
            dataset
            .sort_by("eval_fp", reverse=True)
            .filter_labels(MAIN, F("eval") == "fp")
        )
        # Visualize results in the App
        session = fo.launch_app(view=view, remote=True, port=5152)
    else:
        # normal view without false positive filters
        session = fo.launch_app(dataset, remote=True, port=5152)
    return session 