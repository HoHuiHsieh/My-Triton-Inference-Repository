from PIL import Image, ImageDraw
from postprocessing import extract_classes_bboxes, transform_bbox_to_original, postprocess_text

classes, bboxes, texts = extract_classes_bboxes(generated_text)
bboxes = [transform_bbox_to_original(bbox, image.width, image.height) for bbox in bboxes]

# Specify output formats for postprocessing
table_format = 'latex' # latex | HTML | markdown
text_format = 'markdown' # markdown | plain
blank_text_in_figures = False # remove text inside 'Picture' class
texts = [postprocess_text(text, cls = cls, table_format=table_format, text_format=text_format, blank_text_in_figures=blank_text_in_figures) for text, cls in zip(texts, classes)]

for cl, bb, txt in zip(classes, bboxes, texts):
    print(cl, ': ', txt)

draw = ImageDraw.Draw(image)
for bbox in bboxes:
  draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline="red")
