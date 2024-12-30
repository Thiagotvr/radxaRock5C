#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from rknn.api import RKNN

def main():
    #--------------------------------------------------
    # 1) Caminho do modelo .rknn (hand_landmark_full)
    #--------------------------------------------------
    rknn_model_path = 'hand_landmark_full.rknn'  
    #--------------------------------------------------
    # 2) Inicia o objeto RKNN e carrega o modelo
    #--------------------------------------------------
    print('[INFO] Carregando o modelo RKNN...')
    rknn = RKNN()
    ret = rknn.load_rknn(rknn_model_path)
    if ret != 0:
        print('[ERRO] Falha ao carregar o modelo RKNN.')
        return

    print('[INFO] Inicializando runtime (NPU)...')
    ret = rknn.init_runtime()
    if ret != 0:
        print('[ERRO] Falha ao inicializar a runtime.')
        return

    #--------------------------------------------------
    # 3) Abre a câmera USB
    #--------------------------------------------------
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('[ERRO] Não foi possível acessar a câmera USB.')
        return

    # Ajusta a resolução (opcional):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    #--------------------------------------------------
    # 4) Loop de captura e inferência em tempo real
    #--------------------------------------------------
    print('[INFO] Iniciando loop de inferência...')
    while True:
        ret, frame = cap.read()
        if not ret:
            print('[ERRO] Falha ao capturar frame da câmera.')
            break

        #----------------------------------------------
        # 4.1) Pré-processamento para o modelo
        #     Ajuste de acordo com o input do hand_landmark_full
        #----------------------------------------------
        # Exemplo: se o modelo espera 224x224, RGB, normalizado
        input_size = (224, 224)
        frame_resized = cv2.resize(frame, input_size)

        # Converte BGR -> RGB, se necessário
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Normalização (dependendo se no conversion script você setou mean=[0,0,0], std=[1,1,1], etc.)
        # Exemplo: se o modelo foi treinado com valores em [0,1], então:
        input_data = frame_rgb.astype(np.float32) / 255.0

        # Gera batch dimension: (1, 224, 224, 3)
        input_data = np.expand_dims(input_data, axis=0)

        #----------------------------------------------
        # 4.2) Execução da inferência
        #----------------------------------------------
        outputs = rknn.inference(inputs=[input_data])

        #----------------------------------------------
        # 4.3) Pós-processamento e exibição
        #----------------------------------------------
        # Supondo que o modelo retorne landmarks (x, y, z) para cada ponto.
        # O shape do output depende de como você converteu o modelo .tflite -> .rknn.
        # Ajuste conforme a saída real do seu modelo.
        # Ex: (1, 63) => 21 landmarks * 3 coords
        # outputs[0] -> shape (1, 63)
        # Pegando esse array e reorganizando:
        landmarks_raw = outputs[0]  # shape: (1, 63)
        landmarks_raw = np.squeeze(landmarks_raw)  # shape: (63,)

        # Se for 21 landmarks, cada um com (x,y,z):
        num_landmarks = 21
        coords = landmarks_raw.reshape(num_landmarks, 3)

        # Vamos desenhar na imagem original (frame)
        # Precisamos "desnormalizar" as coordenadas se o modelo espera [0,1] relativo a 224x224
        # ou se o próprio hand_landmark faz a normalização interna. 
        # A maneira correta depende de como o modelo foi treinado.
        # Aqui, como exemplo, vou assumir que as coords (x,y) são *relativas* ao 224x224:
        for i in range(num_landmarks):
            x = coords[i][0] * input_size[0]  # recupera para o tamanho 224
            y = coords[i][1] * input_size[1]
            z = coords[i][2]  # pode ser a profundidade normalizada

            # Desenha um ponto na imagem redimensionada
            cv2.circle(frame_resized, (int(x), int(y)), 2, (0,255,0), -1)

        # (Opcional) Se quiser ver esses pontos no frame original 640x480,
        # você precisaria mapear as coordenadas. Mas para demonstração,
        # vamos só exibir no frame redimensionado:
        cv2.imshow('Hand Landmark (resized)', frame_resized)

        # Se quiser exibir o frame original:
        cv2.imshow('Camera Original', frame)

        #----------------------------------------------
        # 4.4) Encerrar com ESC ou 'q'
        #----------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC ou 'q'
            break

    #--------------------------------------------------
    # 5) Libera recursos
    #--------------------------------------------------
    cap.release()
    rknn.release()
    cv2.destroyAllWindows()
    print('[INFO] Finalizado com sucesso!')

if __name__ == '__main__':
    main()
