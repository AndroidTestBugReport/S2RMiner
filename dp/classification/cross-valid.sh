     for crosstimes in {1..30}
     do
        echo "The value is: $crosstimes"
        python3 randomBuild_crossValid.py
        sleep 0.1m
        python3 Keras_embedding_word2vec_CNN-LSTm-crossValidation.py
     done
