# MyClassifierModel Usage

```
import org.tensorflow.lite.classify.MyClassifierModel;

// 1. Initialize the Model
MyClassifierModel model = null;

try {
    model = new MyClassifierModel(context);  // android.content.Context
    // Create the input container.
    MyClassifierModel.Inputs inputs = model.createInputs();
} catch (IOException e) {
    e.printStackTrace();
}

if (model != null) {

    // 2. Set the inputs
    // Load input tensor "image" from a Bitmap with ARGB_8888 format.
    Bitmap bitmap = ...;
    inputs.loadImage(bitmap);
    // Alternatively, load the input tensor "image" from a TensorImage.
    // Check out TensorImage documentation to load other image data structures.
    // TensorImage tensorImage = ...;
    // inputs.loadImage(tensorImage);

    // 3. Run the model
    MyClassifierModel.Outputs outputs = model.run(inputs);

    // 4. Retrieve the results
    Map<String, Float> probability = outputs.getProbability();
}
```
