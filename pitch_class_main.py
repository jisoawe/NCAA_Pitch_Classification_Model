from pitch_class_etl import load_data, clean_data, transform_data, save_data
from pitch_class_model_training import split_data, train_model
from pitch_class_model_eval import evaluate_model
from pitch_class_predictions import make_predictions, load_new_data

def main():
    data = load_data('/Pitch_Classification_Model24/NCAA_pitch.csv')

    transformed_data = transform_data(data)
    cleaned_data = clean_data(transformed_data)
    save_data(cleaned_data, '/Pitch_Classification_Model24/processed_data.csv')

    X_train, X_test, y_train, y_test = split_data(cleaned_data, target_column = 'TaggedPitchType')
    model = train_model(X_train, y_train)

    model_evaluation = evaluate_model(model, X_test, y_test)
    print(model_evaluation)

    # for loading new data
    new_data = load_new_data('/Pitch_Classification_Model24/new_data.csv')
    
    new_transformed_data = transform_data(new_data)
    new_cleaned_data = clean_data(new_transformed_data)
    save_data(new_cleaned_data, '/Pitch_Classification_Model24/new_processed_data.csv')

    X_train, X_test, y_train, y_test = split_data(new_cleaned_data, target_column = 'TaggedPitchType')
    model2 = train_model(X_train, y_train)

    new_cleaned_features = new_cleaned_data.drop(columns=['TaggedPitchType'])
    predictions = make_predictions(model2, new_cleaned_features)
    save_data(predictions, '/Pitch_Classification_Model24/predicted_data.csv')

    model_evaluation = evaluate_model(model2, X_test, y_test)
    print(model_evaluation)

if __name__ == '__main__':
    main()


