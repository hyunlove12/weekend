from keras import layers, models, datasets
#전처리
from keras.preprocessing import sequence

class Data:
    def  __init__(self, max_features=200000, maxlen=80):
        #test가 있으며 지도학습(답을 정한 것)
        (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(
            num_words=max_features )
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

#model은 완성되어 있다.
#rnn은 텍스트, cnn은 이미지지
#lstm은 타임라인 -> 정확도가 높아진다(시간차)
#recurrent_dropout -> 순환
#dense, drop에 lstm(시간이 지나도 기록)추가
class RNN_LSTM(models.Model):
    def __init__(self, max_features, maxlen):
        #(()) 2번 사용 시 튜플 타입
        x = layers.Input((maxlen,))
        #rnn
        h = layers.Embedding(max_features, 128)(x)
        h = layers.LSTM(128,
                        dropout=0.2,
                        recurrent_dropout=0.2)(h)
        y = layers.Dense(1, activation='sigmoid')(h)
        super().__init__(x, y)
        #배표
        self.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

class Machine:
    def __init__(self, max_features=20000, maxlen=80):
        #데이터 만드는 것 하나, 모델링 하는 것 하나
        #데이터를 모델에 넣어서 공부
        self.data = Data(max_features, maxlen)
        self.model = RNN_LSTM(max_features, maxlen)

    #epochs 반복 횟수
    #batch-size=32 기존 함수에 선언된 변수와 이름이 똑같아야 non-defalut parameter
    def run(self, epochs=3, batch_size=32):
        data = self.data
        model = self.model
        print('Training State')
        model.fit(
            data.x_train,
            data.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(data.x_test,
                             data.y_test),
            verbose=2
        )
        loss, acc = model.evaluate(
            data.x_test,
            data.y_test,
            batch_size=batch_size,
            verbose=2
        )
        print('Test performance accuracy={0}, loss{1}'.format(acc, loss))

if __name__ == '__main__':
    m = Machine()
    m.run()





























