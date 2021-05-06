from lib import model

if __name__ == '__main__':
    model_params = model.ModelParams(
        base_model=model.BaseModel.RESNET50_v2,
        image_size=160,
        num_classes=2,
        dropout=0.2,
        train_layers=2,
        global_pooling=model.GlobalPooling.AVG_POOLING,
        use_data_augmentation=True
    )
