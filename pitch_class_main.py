from pitch_class_etl import load_data, clean_data, transform_data, save_data
from pitch_class_model_training import split_data, train_model
from pitch_class_model_eval import evaluate_model
from pitch_class_predictions import make_predictions, load_new_data

def main():
    data = load_data('NCAA_Pitch.csv')

    transformed_data = transform_data(data)
    cleaned_data = clean_data(transformed_data)
    save_data(cleaned_data, 'processed_data.csv')

    X_train, X_test, y_train, y_test = split_data(transformed_data, target_column = 'TaggedPitchType')
    model = train_model(X_train, y_train)

    model_evaluation = evaluate_model(model, X_test, y_test)
    print(model_evaluation)

    # for loading new data types
    new_data = load_new_data('new_data.csv')
    
    predictions = make_predictions(model, new_data)
    print(predictions)

if __name__ == '__main__':
    main()


