Data Visualization:

1. implementation block diagram
2. Convolutional Neural Net schemas:
    1. LeNet5
    2. DenseNet120
    - compare n params
3. Images of results
    0. Data distribution - column chart
    1. Test Error (%) - with/without data aug - column chart
    2. Test Error x Epoch - with data aug - line plot
    3. Time per epoch and total training time - table
    
    - obs:
        - compare same resolution
        - compare same aug params
        - compare same k-fold splits
        - compare qualitatively: train set x test set
            - images it does not get
            - images it gets
        
Final chars:
    - 5 folds
    - 1024 aug images (from 470 * 4 / 5) for training
    - 512 batch size
    - 100 epochs
        
