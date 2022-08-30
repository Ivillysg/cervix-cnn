Linguagem Utilizada: Python 3.8
Bibliotecas: Keras, OpenCV, Sklearn, numpy, matplotlib.


No arquivo settings.py, estão todas as configurações globais da aplicação, desde tamanho das imagens até nome das pastas, quantidade de épocas para treinamento, etc. Antes de começar o treinamento, certifique-se que tenham uma pasta no local raiz com o nome images contendo as fotos a serem utilizadas para treinamento.

Logo depois, executar o arquivo main.py, sendo ele responsável por chamar a função de preparação, que servirá para redimensionar e filtrar, assim como chamar a rede para treinamento. Assim que iniciado o arquivo, irá verificar se existe uma pasta chamada INPUT, se não existir, será chamada a função resizedImages, responsável por criar a pasta, redimensionar e filtrar as imagens. Caso contrário, se já existir a pasta INPUT, então as imagens contidas, serão pré-processadas, e carregadas em memória, para posteriormente serem utilizadas no treinamento da rede.

Ademais, a rede irá começar o treinamento, contendo o número de épocas que foram definidas no arquivo Settings.py. Quando terminado o treinamento, a mesma irá salvar dois arquivo de imagens, chamados de predict.png e gráfico.png, contendo as predições e os gráficos de validação e treino, respectivamente.

Obs: para serem feitas as predições corretamente, será necessário uma pasta chamada test, contendo imagens que não foram utilizadas para o treinamento da rede.

Lembrando, o mesmo código poderá ser reutilizado para predizer qualquer tipo de classificação, desde que seja pré configurado de acordo com o objetivo final.

Obs: As imagens contidas na pasta IMAGES, deverão ter suas categorias pre definidas para o funcionamento do código.
Exemplo:

Na figura, temos 3 categorias distintas.
Todo o conhecimento utilizado, foi através de conteúdo da internet, youtube e Udemy(cursos pagos).