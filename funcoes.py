#Alan Costa Bráulio
#Igor da Costa Silva
#Natália Batista Oliveira

import cv2  #manipular imagem
import numpy as np #arrays
import tkinter as tk #interface gráfica
from tkinter import *
from tkinter import filedialog
import os #acessar diretório
from matplotlib import pyplot #plotar gráficos
from PIL import Image, ImageTk #interface gráfica
import glob #acessar diretorio
import mahotas as mt #haralick
import sklearn #rede neural
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import confusion_matrix #calcular matriz de confusão
import matplotlib.pyplot as plt  
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix

class Imagem:
    def __init__(self):
        self.imgoriginal = np.zeros
        self.imgatual = np.zeros
        self.imganterior = np.zeros
        self.imgtemporaria = np.zeros
        

    def buscarimg(self):
        root = tk.Tk() 
        root.withdraw()
        filename = filedialog.askopenfilename(title = 'Select File', filetypes = [("PNG File", "*.png"), ("TIFF File", "*.tiff")])
        self.caminhodoarquivo = filename
        img = cv2.imread(filename)
      #  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       # small = cv2.resize(img, (0,0), fx=0.6, fy=0.6) 
        self.imgoriginal = img
        self.imgatual = img
        self.imgtemporaria = img
        cv2.imwrite("imagematual.png", img)    
        cv2.imshow('Imagem',self.imgatual)
       # cv2.imwrite("imagematual.png", self.imgatual)
        
       # cv2.waitKey() 
       

    def zoomin(self):
        cv2.destroyAllWindows()
       # self.imgtemporaria = cv2.resize(self.imgtemporaria, (0,0), fx=1.05, fy=1.05) 
        im = self.imgtemporaria
        height, width = im.shape[:2]
        thumbnail = cv2.resize(im, (round(width / 0.9), round(height / 0.9)), interpolation=cv2.INTER_AREA)
        cv2.imshow('Imagem', thumbnail)
        self.imgtemporaria = thumbnail
      #  cv2.imshow('Imagem',self.imgtemporaria)

    def zoomout(self):
        cv2.destroyAllWindows()
       # self.imgtemporaria = cv2.resize(self.imgtemporaria, (0,0), fx=1.05, fy=1.05) 
        im = self.imgtemporaria
        height, width = im.shape[:2]
        thumbnail = cv2.resize(im, (round(width / 1.1), round(height / 1.1)), interpolation=cv2.INTER_AREA)
        cv2.imshow('Imagem', thumbnail)
        self.imgtemporaria = thumbnail

    def exibirImagemOriginal(self):
        cv2.destroyAllWindows()
        cv2.imshow('Imagem',self.imgoriginal)
        self.imgtemporaria = self.imgatual

    def exibirImagemAtual(self):
        cv2.destroyAllWindows()
        cv2.imshow('Imagem',self.imgatual)
        self.imgtemporaria = self.imgatual
    
    def desfazerAlteracao(self):
        cv2.destroyAllWindows()
        self.imgatual = self.imganterior
        self.imgtemporaria = self.imganterior
        cv2.imshow('Imagem',self.imgatual)

    #Selecionar 128x128 FALTA MUDAR PARA RECORTAR APENAS 128X128
    def selecionar(self):
        cv2.destroyAllWindows()
        # Read image
        im = self.imgtemporaria
        showCrosshair = False 
        fromCenter    = False
        # Select ROI
        myroi = cv2.selectROI("Imagem", im, fromCenter, showCrosshair)
        
        # Crop image
        imCrop = im[int(myroi[1]):int(myroi[1]+myroi[3]), int(myroi[0]):int(myroi[0]+myroi[2])]
        cv2.destroyAllWindows()
        # Display cropped image
        self.imganterior = self.imgatual 
        self.imgatual = imCrop
        self.imgtemporaria = self.imgatual
        cv2.imshow("Imagem", self.imgatual)
        
        #cv2.waitKey(0)
        #cv2.imwrite("imgName.jpg", imCrop)

    #Transformar para 64x64
    def resize64(self):
        cv2.destroyAllWindows()
        img = self.imgatual
        tamanho_novo = (64, 64)
        img_redimensionada = cv2.resize(img, 
        tamanho_novo, interpolation = cv2.INTER_AREA) 
        cv2.imshow('Imagem', img_redimensionada)
        self.imganterior = self.imgatual 
        self.imgatual = img_redimensionada
        self.imgtemporaria = self.imgatual
        cv2.imwrite("imagematual.png", self.imgatual)
        

    #Transformar para 32x32
    def resize32(self):
        cv2.destroyAllWindows()
        img = self.imgatual
        tamanho_novo = (32, 32)
        img_redimensionada = cv2.resize(img, 
        tamanho_novo, interpolation = cv2.INTER_AREA) 
        cv2.imshow('Imagem', img_redimensionada)
        self.imganterior = self.imgatual 
        self.imgatual = img_redimensionada
        self.imgtemporaria = self.imgatual
       
    
    #ALGORITMO DE QUANTIZAÇÃO 256, 32, 16
    def quantizacao256(self):
        cv2.destroyAllWindows()
        ## Quantização ##
        img = self.imgatual
        r = 1
        imgQuant = np.uint8(img / r) * r
        self.imganterior = self.imgatual 
        self.imgatual = imgQuant
        self.imgtemporaria = self.imgatual
        cv2.imshow('Imagem', self.imgatual)
        
    
    def quantizacao32(self):
        ## Quantização ##
        cv2.destroyAllWindows()
        img = self.imgatual
        r = 8
        imgQuant = np.uint8(img / r) * r
        self.imganterior = self.imgatual 
        self.imgatual = imgQuant
        self.imgtemporaria = self.imgatual
        cv2.imshow('Imagem', self.imgatual)
        
    
    def quantizacao16(self):
        ## Quantização ##
        cv2.destroyAllWindows()
        img = self.imgatual
        r = 16
        imgQuant = np.uint8(img / r) * r
        self.imganterior = self.imgatual 
        self.imgatual = imgQuant
        self.imgtemporaria = self.imgatual
        cv2.imshow('Imagem', self.imgatual)
        

    #EQUALIZAÇÃO
    def calcular_histograma(self):
        imagem = self.imgatual
        histograma = cv2.calcHist([imagem], [0], None, [256], [0, 256])
        pyplot.plot(histograma)
        pyplot.show()
   
    def equalize(self):
        cv2.destroyAllWindows()
        img = self.imgatual
        self.calcular_histograma()
        img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
        img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
        hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
        self.imganterior = self.imgatual 
        self.imgatual = hist_equalization_result
        self.imgtemporaria = self.imgatual
        cv2.imshow('Imagem',self.imgatual)
        self.calcular_histograma()
        
    def extract_features(self,image):
        
        textures = mt.features.haralick(image)

        ht_mean = textures.mean(axis=0)
        return ht_mean

    def classificar(self):

        train_path = "dataset/train"
        train_names = os.listdir(train_path)

        X = []
        Y = []

        for train_name in train_names:
            cur_path = train_path + "/" + train_name
            cur_label = train_name
            i = 1
            for file in glob.glob(cur_path + "/*.png"):
                print ("Processing Image - {} in {}".format(i, cur_label))
               
                image = cv2.imread(file)

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resize64 = cv2.resize(gray,(64,64), interpolation = cv2.INTER_AREA) 
                resize32 = cv2.resize(resize64,(32,32), interpolation = cv2.INTER_AREA) 
                #quantização
                Quant128p32 = np.uint8(gray / 8) * 8
                Quant128p16 = np.uint8(gray / 16) * 16
                Quant64p32 = np.uint8(resize64 / 8) * 8
                Quant64p16 = np.uint8(resize64 / 16) * 16
                Quant32p32 = np.uint8(resize32 / 8) * 8 
                Quant32p16 = np.uint8(resize32 / 16) * 16
                
                
                #extract haralick texture ORIGINAL IMAGEM
                features = self.extract_features(gray)
                features64 = self.extract_features(resize64)
                features32 = self.extract_features(resize32)
                features128p32 = self.extract_features(Quant128p32)
                features128p16 = self.extract_features(Quant128p16)
                features64p32 = self.extract_features(Quant64p32)
                features64p16 = self.extract_features(Quant64p16)
                features32p32 = self.extract_features(Quant32p32)
                features32p16 = self.extract_features(Quant32p16)
                          
                
                X.append(features)
                Y.append(cur_label)

                X.append(features64)
                Y.append(cur_label)

                X.append(features32)
                Y.append(cur_label)

                X.append(features128p32)
                Y.append(cur_label)

                X.append(features128p16)
                Y.append(cur_label)

                X.append(features64p32)
                Y.append(cur_label)

                X.append(features128p16)
                Y.append(cur_label)

                X.append(features32p32)
                Y.append(cur_label)

                X.append(features32p16)
                Y.append(cur_label)

               
                i += 1

      
        print ("Training features: {}".format(np.array(X).shape))
        print ("Training labels: {}".format(np.array(Y).shape))


        #VALIDAÇÃO CRUZADA
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
        #print(len(X_train))
        #print(len(X_test))

        clf = LinearSVC(random_state=9, dual=False, max_iter=5000)
        clf.fit(X_train,y_train) #treina a rede
        scores = clf.score(X_test,y_test) #calcula o score
        scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy') 
        score = scores.mean()
        print(score)
        y_test = np.reshape(y_test,(-1, 1))
        plot_confusion_matrix(clf, X_test, y_test)  
        plt.show()  
    
        # convert to grayscal
        conv = cv2.cvtColor(self.imgatual, cv2.COLOR_BGR2GRAY)
        # extract haralick texture from the image
        features = self.extract_features(conv)
        # evaluate the model and predict label
        prediction = clf.predict(features.reshape(1, -1))[0]
        # show the label
        cv2.putText(conv, prediction, (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 3)
        
        # display the output image
        cv2.destroyAllWindows()
        cv2.imshow("Imagem", conv)
        cv2.waitKey(0)  

#INTERFACE GRÁFICA

    def novaTela(self):
        #MAIN FRAME
        app=Tk()
        app.title("RECONHECIMENTO DE BRADS")
        app.configure(background="#D3D3D3")
        #app.iconbitmap("icon.ico")
        app.geometry("350x400")
        app.resizable(False,False)

        
        #MENU
        barraDeMenus = Menu(app)

        #MENU ARQUIVO
        menuArquivo= Menu(barraDeMenus,tearoff=0)
        menuArquivo.add_command(label="Abrir",command=self.buscarimg)
        menuArquivo.add_command(label="Testar Birad",command=self.classificar)
        menuArquivo.add_separator()
        menuArquivo.add_command(label="Fechar",command=app.quit)
        #MENU REDIMENCIONAR
        menuRedmencionar= Menu(barraDeMenus,tearoff=0)
        menuRedmencionar.add_command(label="Selecionar Área",command=self.selecionar)
        menuRedmencionar.add_command(label="64x64",command=self.resize64)
        menuRedmencionar.add_command(label="32x32",command=self.resize32)
        #MENU QUANTIZAÇÃO
        menuQuantizar= Menu(barraDeMenus,tearoff=0)
        menuQuantizar.add_command(label="256",command=self.quantizacao256)
        menuQuantizar.add_command(label="32",command=self.quantizacao32)
        menuQuantizar.add_command(label="16",command=self.quantizacao16)
        #MENUEQUALIZAR
        menuEqualizar= Menu(barraDeMenus,tearoff=0)
        menuEqualizar.add_command(label="Equalização",command=self.equalize)

        #BARRA DE MENUS
        barraDeMenus.add_cascade(label="Arquivo",menu=menuArquivo)
        barraDeMenus.add_cascade(label="Redimencionar",menu=menuRedmencionar)
        barraDeMenus.add_cascade(label="Quantização",menu=menuQuantizar)
        barraDeMenus.add_cascade(label="Equalização",menu=menuEqualizar)
        
        #Labels
        app.config(menu=barraDeMenus)
         
        #CABECALHO FRAME
        cabecalhoFrame = tk.LabelFrame(app,bd=0)
        cabecalhoFrame.place(relwidth=1,relheight=0.10)

        txt1=Label(cabecalhoFrame,text="Reconhecimento de BI-RADS",background="#000080", fg="#fff",
        font="Roboto 15 bold" )
        txt1.place(relwidth=1,relheight=1)

        #ENTRADA FRAME
        entradaFrame = tk.LabelFrame(app,bd=2)
        entradaFrame.place(relwidth=1,relheight=0.07,rely=0.11)
        
        
        btnCarregar = tk.Button(entradaFrame,bd=0,font="Roboto 10 bold",text="Buscar Imagem", command=self.buscarimg)
        btnCarregar.place(relwidth=1,relheight=1)

        #ZOOM FRAME
        zoomFrame = tk.LabelFrame(app,background="#DCDCDC",bd=0)
        zoomFrame.place(relwidth=1,relheight=0.07,rely=0.20)

        btnzoomin = tk.Button(zoomFrame,bd=0,font="Roboto 10 bold",text="ZOOM IN", command=self.zoomin)
        btnzoomin.place(relwidth=0.25,relheight=1, relx=0.25)

        btnzoomout = tk.Button(zoomFrame,bd=0,font="Roboto 10 bold",text="ZOOM OUT", command=self.zoomout)
        btnzoomout.place(relwidth=0.25,relheight=1,relx=0.55)

        #IMAGEM FRAME
        imagemFrame = tk.LabelFrame(app,background="#DCDCDC",bd=0)
        imagemFrame.place(relwidth=1,relheight=0.15,rely=0.30)

        btnImgOriginal = tk.Button(imagemFrame,bd=0,font="Roboto 10 bold",text="Imagem\n Original", command=self.exibirImagemOriginal)
        btnImgOriginal.place(relwidth=0.30,relheight=1)

        btnImgAtual = tk.Button(imagemFrame,bd=0,font="Roboto 10 bold",text="Imagem\n Atual", command=self.exibirImagemAtual)
        btnImgAtual.place(relwidth=0.30,relheight=1,relx=0.35)

        btnDesfazer = tk.Button(imagemFrame,bd=0,font="Roboto 10 bold",text="Desfazer \n Última Alteração", command=self.desfazerAlteracao)
        btnDesfazer.place(relwidth=0.30,relheight=1,relx=0.70)

        #DIMENCAO FRAME
        dimencaoFrame = tk.LabelFrame(app,background="#DCDCDC",bd=0)
        dimencaoFrame.place(relwidth=1,relheight=0.10, rely=0.47)

        btnresize128 = tk.Button(dimencaoFrame,bd=0,font="Roboto 10 bold",text="Selecionar Área", command=self.selecionar)
        btnresize128.place(relwidth=0.30,relheight=1)

        btnresize64 = tk.Button(dimencaoFrame,bd=0,font="Roboto 10 bold",text="64x64", command=self.resize64)
        btnresize64.place(relwidth=0.30,relheight=1,relx=0.35)

        btnresize32 = tk.Button(dimencaoFrame,bd=0,font="Roboto 10 bold",text="32x32", command=self.resize32)
        btnresize32.place(relwidth=0.30,relheight=1,relx=0.70)

        #QUANTIZACAO FRAME
        quantizacaoFrame = tk.LabelFrame(app,background="#DCDCDC",bd=0)
        quantizacaoFrame.place(relwidth=1,relheight=0.10,rely=0.60)

        btnQuant256 = tk.Button(quantizacaoFrame,bd=0,font="Roboto 10 bold",text="256", command=self.quantizacao256)
        btnQuant256.place(relwidth=0.30,relheight=1)

        btnQuant32 = tk.Button(quantizacaoFrame,bd=0,font="Roboto 10 bold",text="32", command=self.quantizacao32)
        btnQuant32.place(relwidth=0.30,relheight=1,relx=0.35)

        btnQuant16 = tk.Button(quantizacaoFrame,bd=0,font="Roboto 10 bold",text="16", command=self.quantizacao16)
        btnQuant16.place(relwidth=0.30,relheight=1,relx=0.70)
        
        #EQUALIZACAO FRAME
        equalizacaoFrame = tk.LabelFrame(app,background="#DCDCDC",bd=0)
        equalizacaoFrame.place(relwidth=0.8,relheight=0.10,rely=0.73,relx=0.10)

        btnEqualizar = tk.Button(equalizacaoFrame,bd=0,font="Roboto 10 bold",text="Equalização", command=self.equalize)
        btnEqualizar.place(relwidth=1,relheight=1)

        #botao FRAME
        TestarFrame = tk.LabelFrame(app,bd=0)
        TestarFrame.place(relwidth=1,relheight=0.13,rely=0.85)

        btnTestar = tk.Button(TestarFrame,bd=0,background="#000080", fg="#fff",font="Roboto 10 bold",text="TESTAR BI-RAD", command=self.classificar)
        btnTestar.place(relwidth=1,relheight=1)
    
    


        app.mainloop()





#MAIN
tela = Imagem()
tela.novaTela()



#img ='imagem1.png'
#tela = Imagem()
#tela.abririmagem(img)
#tela.recortar()
#tela.buscarimg()
#tela.cortar()
#tela.selecionar()
#tela.resize64()
#tela.quantizacao32()
#tela.calcular_histograma()
#tela.equalize()
