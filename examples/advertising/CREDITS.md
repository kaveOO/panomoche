# Example images and model data

- `landscape.jpg`: **Fronalpstock, Switzerland**, photographed by **Hannes Röst**. Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Fronalpstock_big.jpg). Used as the 1280-pixel thumbnail. Licensed [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
- `advertisement-panorama.png`: synthetic test advertisement made by cropping/resizing the above landscape and adding promotional text and a graphic border. This derivative image is also [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The advertisement is fictional.
- `street.jpg`, `street-wavy.png` and `animals.png`: copies of images supplied by the user in this conversation for testing.
- `street-with-sign.jpg`: a supplied street image with a small synthetic sale sign added as a controlled regression fixture.
- `advertisement-with-street.png`: a fictional advertising layout incorporating the supplied street photograph. It tests whether scenery inside an advertisement bypasses the filter.
- `screenshot.jpg`: a screenshot of the previously created local waviness tool, used as a non-advertising regression fixture.
- `advertisement.png`, `document.png`, `logo.png` and `blank.png`: synthetic test fixtures generated locally. Text and logos are fictional.
- `park-statue.jpg` and `ice-landscape.jpg`: copies of the two additional photographs supplied by the user as valid map imagery.
- `advertisement-with-statue.png` and `advertisement-with-ice.png`: fictional promotional layouts incorporating those supplied photographs as negative regression controls.
- `portrait.jpg`: NASA portrait photograph **S69-31743**, obtained as a 960-pixel thumbnail from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Buzz_Aldrin.jpg). The source lists it as public-domain NASA material; the current restoration is credited there to Coffeeandcrumbs. It is used only as a non-advertising portrait regression example, with no endorsement implied.

The examples are development fixtures, not a production training or validation dataset.

The model is Google's **SigLIP 2 Base**, using the quantized ONNX export published by the Hugging Face ONNX Community at revision `ba1f3b0843f24bc5417d38e19c37b287d719b2f4`. Model data is [Apache 2.0 licensed](https://www.apache.org/licenses/LICENSE-2.0). The model binaries are downloaded and verified during setup, and are excluded from the source ZIP.

- `advertisement-restaurant.png` and `advertisement-resort.png`: real advertising graphics supplied by the user as failed detection examples. Included for local regression testing; no ownership or redistribution license is asserted. These images are not training data.
