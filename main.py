# ライブラリのインポート
import streamlit as st
import numpy as np
import cv2
#import io
#import os
#import json
import torch
import pytorch_lightning as pl
from torchvision.models import resnet18, resnet50, efficientnet_b0, efficientnet_b3
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tempfile
from pathlib import Path
import pandas as pd
#from azure.cognitiveservices.vision.face import FaceClient
#from msrest.authentication import CognitiveServicesCredentials


# Face APIの各種設定
# jsonファイルを読み込む
#with open('secret.json') as f:
 # secret_json = json.load(f)

#subscription_key = secret_json['AZURE_KEY'] # AzureのAPIキー
#endpoint = secret_json['AZURE_URL'] # AzureのAPIエンドポイント

# キーが無ければ強制終了
#assert subscription_key

# クライアントの認証
#face_client = FaceClient(endpoint, CognitiveServicesCredentials(subscription_key))


# 各関数の定義
# モデルを読み込む関数
# @st.casheで再読み込みにかかる時間を減らす。
#@st.cache(allow_output_mutation=True)

class LitNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = efficientnet_b0(pretrained=False)

        
        self.fc_labels = []

        self.fc_1_1 = nn.Linear(1000, 2); self.fc_labels.append(self.fc_1_1)
        self.fc_1_2 = nn.Linear(1000, 2); self.fc_labels.append(self.fc_1_2)
        self.fc_1_3 = nn.Linear(1000, 2); self.fc_labels.append(self.fc_1_3)
        self.fc_1_4 = nn.Linear(1000, 2); self.fc_labels.append(self.fc_1_4)
        self.fc_2_1 = nn.Linear(1000, 2); self.fc_labels.append(self.fc_2_1)
        self.fc_2_2 = nn.Linear(1000, 2); self.fc_labels.append(self.fc_2_2)
        self.fc_2_3 = nn.Linear(1000, 2); self.fc_labels.append(self.fc_2_3)
        self.fc_2_4 = nn.Linear(1000, 2); self.fc_labels.append(self.fc_2_4)
        self.fc_2_5 = nn.Linear(1000, 2); self.fc_labels.append(self.fc_2_5)
        self.fc_2_6 = nn.Linear(1000, 3); self.fc_labels.append(self.fc_2_6)
        self.fc_3_1 = nn.Linear(1000, 2); self.fc_labels.append(self.fc_3_1)
        self.fc_3_2 = nn.Linear(1000, 2); self.fc_labels.append(self.fc_3_2)
        self.fc_3_3 = nn.Linear(1000, 2); self.fc_labels.append(self.fc_3_3)
        self.fc_3_4 = nn.Linear(1000, 3); self.fc_labels.append(self.fc_3_4)
        self.fc_3_5 = nn.Linear(1000, 3); self.fc_labels.append(self.fc_3_5)
        self.fc_4_1 = nn.Linear(1000, 2); self.fc_labels.append(self.fc_4_1)
        self.fc_4_2 = nn.Linear(1000, 5); self.fc_labels.append(self.fc_4_2)
        self.fc_4_3 = nn.Linear(1000, 5); self.fc_labels.append(self.fc_4_3)
        self.fc_5_1 = nn.Linear(1000, 2); self.fc_labels.append(self.fc_5_1)
        self.fc_5_2 = nn.Linear(1000, 5); self.fc_labels.append(self.fc_5_2)
        self.fc_5_3 = nn.Linear(1000, 5); self.fc_labels.append(self.fc_5_3)
        self.fc_5_4 = nn.Linear(1000, 5); self.fc_labels.append(self.fc_5_4)
        self.fc_5_5 = nn.Linear(1000, 5); self.fc_labels.append(self.fc_5_5)
        self.fc_5_6 = nn.Linear(1000, 5); self.fc_labels.append(self.fc_5_6)
        self.fc_5_7 = nn.Linear(1000, 5); self.fc_labels.append(self.fc_5_7)

        self.fc_vas = nn.Linear(1000, 3)

        self.cel = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        feature = self.model(x)
        loss = 0
        if int(exp)==1:
            y_hats = []
            for i in range(14):
                y_hats.append(self.fc_labels[i](feature))
            for i, y_hat in enumerate(y_hats):
                loss = loss + self.cel(y_hat, y[:, i])
        elif int(exp)==2:
            y_hats = []
            for i in range(14, 25):
                y_hats.append(self.fc_labels[i](feature))
            for i, y_hat in enumerate(y_hats):
                loss = loss + self.cel(y_hat, y[:, i])
        elif int(exp)==3:
            y_hat = self.fc_vas(feature)
            for i in range(3):
                loss = loss + self.mse(y_hat[:, i], y[:, i])  
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        feature = self.model(x)
        loss = 0
        if int(exp)==1:
            y_hats = []
            for i in range(14):
                y_hats.append(self.fc_labels[i](feature))
            for i, y_hat in enumerate(y_hats):
                loss = loss + self.cel(y_hat, y[:, i])
        elif int(exp)==2:
            y_hats = []
            for i in range(14, 25):
                y_hats.append(self.fc_labels[i](feature))
            for i, y_hat in enumerate(y_hats):
                loss = loss + self.cel(y_hat, y[:, i])
        elif int(exp)==3:
            y_hat = self.fc_vas(feature)
            for i in range(3):
                loss = loss + self.mse(y_hat[:, i], y[:, i])  
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        feature = self.model(x)
        loss = 0
        if int(exp)==1:
            y_hats = []
            for i in range(14):
                y_hats.append(self.fc_labels[i](feature))
            for i, y_hat in enumerate(y_hats):
                loss = loss + self.cel(y_hat, y[:, i])
        elif int(exp)==2:
            y_hats = []
            for i in range(14, 25):
                y_hats.append(self.fc_labels[i](feature))
            for i, y_hat in enumerate(y_hats):
                loss = loss + self.cel(y_hat, y[:, i])
        elif int(exp)==3:
            y_hat = self.fc_vas(feature)
            for i in range(3):
                loss = loss + self.mse(y_hat[:, i], y[:, i])  
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-3)
        return optimizer
    
def model_init(exp):
    """
    モデルの初期化関数
    Parameters
    ----------
    weight_path : str
        学習済みモデルの重みのパス
    exp : str
        ページ番号（1, 2, 3）

    Returns
    -------
    G : pl.LightningModule
        学習済みモデル
    """

    if int(exp)==1:
        #model = LitNet.load_from_checkpoint("saved_model/2023_03_23_12_40_Page_1_(640, 880)_efficientnet_b0/0/epoch=155-val_loss=0.002.ckpt")
        model = LitNet.load_from_checkpoint("epoch=155-val_loss=0.002.ckpt")
    elif int(exp)==2:
        #model = LitNet.load_from_checkpoint("saved_model/2023_03_24_10_42_Page_2_(640, 880)_efficientnet_b0/0/epoch=72-val_loss=1.006.ckpt")
        model = LitNet.load_from_checkpoint("epoch=72-val_loss=1.006.ckpt")
    elif int(exp)==3:
        #model = LitNet.load_from_checkpoint("saved_model/2023_03_22_08_39_Page_3_(640, 880)_efficientnet_b0/0/AI-ISSG-epoch=168-val_loss=1.466.ckpt")
        model = LitNet.load_from_checkpoint("AI-ISSG-epoch=168-val_loss=1.466.ckpt")
    model.eval()

    return model

# preprocess
def preprocess(path):
    transform_val = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])
    image = cv2.imread(str(path))
    image = np.array(cv2.resize(image, (640, 880)))
    data = transform_val(image=image)['image']
    return image, data.unsqueeze(dim=0)

def main():
    # タイトル
    st.title('JOABPEQ AI')

    # サイドバー
    st.sidebar.title('Inference')
    st.sidebar.write('Uproad an image')
    st.sidebar.write('Results are displayed on the right.')
    st.sidebar.write('--------------')
    uploaded_file = st.sidebar.file_uploader("Uproad your image", type=['jpg','jpeg', 'png'])
    # セットアップの定義
    st.sidebar.subheader('Setup')
    exp = st.sidebar.selectbox('The number of page', ('1', '2', "3"))
    # 以下ファイルがアップロードされた時の処理
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            progress_message = st.empty()
            progress_message.write('Now processing. Please wait.')
            fp = Path(tmp_file.name)
            fp.write_bytes(uploaded_file.getvalue())

            G = model_init(exp)
            image, data = preprocess(tmp_file.name)

            with torch.no_grad():
                cols = ['1_1', '1_2', '1_3', '1_4', '2_1', '2_2', '2_3', '2_4', '2_5', '2_6',
            '3_1', '3_2', '3_3', '3_4', '3_5', '4_1', '4_2', '4_3', '5_1', '5_2',
            '5_3', '5_4', '5_5', '5_6', '5_7']
                result = []
                m = nn.Softmax(dim=1)
                if int(exp)==1:
                    result_dict = {}
                    for i in range(14):
                        feature = G.model(data)
                        fc = G.fc_labels[i]
                        pred = np.argmax(m(fc(feature)).detach().cpu().numpy().squeeze())+1
                        value = np.max(m(fc(feature)).detach().cpu().numpy().squeeze())
                        #tmp = f"The prediction label of {cols[i]} is {pred}. Prediction probability is {value}. Probability greater than 0.9996 is extremely reliable"
                        if value>=0.9996:
                            tmp = f"{cols[i]} : {pred}_{value}"
                        else:
                            tmp = f"{cols[i]} : {pred}_{value}"
                        result.append(tmp)
                        result_dict[cols[i]] = pred
                        result_dict[cols[i]+"_pred_prob"] = value
                    df_result = pd.DataFrame(result_dict)
                elif int(exp)==2:
                    result_dict = {}
                    for i in range(14, 25):
                        feature = G.model(data)
                        fc = G.fc_labels[i]
                        pred = np.argmax(m(fc(feature)).detach().cpu().numpy().squeeze())+1
                        value = np.max(m(fc(feature)).detach().cpu().numpy().squeeze())
                        #tmp = f"The prediction label of {cols[i]} is {pred}. Prediction probability is {value}. Probability greater than 0.9996 is extremely reliable"
                        if value>=0.9996:
                            tmp = f"{cols[i]} : {pred}_{value}"
                        else:
                            tmp = f"{cols[i]} : {pred}_{value}"
                        result.append(tmp)
                        result_dict[cols[i]] = pred
                        result_dict[cols[i]+"_pred_prob"] = value
                    df_result = pd.DataFrame(result_dict)
                elif int(exp)==3:
                    feature = G.model(data)
                    result = G.fc_vas(feature).detach().cpu().numpy().squeeze()
                    df_result = pd.DataFrame(result)
                    df_result.columns = ["VAS_1", "VAS_2", "VAS_3"]
            # 元の画像に長方形と名前が書かれているので、それを表示
            #st.image(image, use_column_width=True)

            # カラムを2列に分ける
            # ※st.beta_columns()じゃないとローカル環境では動かないです。
            # ただ本番環境にデプロイするとそれじゃ古すぎる、新しいのに変更しましょうといったアラートが 
            # 出てきたので完成版のコードはst.columns()とこの後のst.expander()にしている
            col1, col2 = st.columns(2)

            # カラム1には検出した顔画像の切り抜きと名前を縦に並べて表示
            with col1:
                st.image(image, use_column_width=True)

            # カラム2には、認識された顔の数だけ上位3人のラベルと確率を表示
            # st.expanderで見たい時にクリックすれば現れるエキスパンダの中に入れる
            with col2:
                st.header('Result')
                if int(exp)==1 or int(exp)==2:
                    for i in range(0, len(result)):
                        st.write(result[i])
                        #st.write(second_name_list[i], 'の可能性:' , round(second_rate_list[i]*100,2), '%')
                        #st.write(third_name_list[i], 'の可能性:' , round(third_rate_list[i]*100,2), '%')
                    st.write("? means unreliable prediction")
                else:
                    for i in range(0, 3):
                        st.write(f"VAS{i+1} is {result[i]}")            
            #@st.cache_data
            def convert_df(df):
               return df.to_csv(index=False).encode('utf-8')
            
            
            csv = convert_df(df_result)
            
            st.download_button(
               "Press to Download",
               csv,
               "file.csv",
               #"text/csv",
               key='download-csv'
            )
            # ここまで処理が終わったら分析が終わったことを示すメッセージを表示
            progress_message.write(f'Finish the inference!')

if __name__ == "__main__":
    main()
