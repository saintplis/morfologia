#include "httplib.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <vector>
#include <mutex>
#include <cstdlib>

using namespace cv;
using namespace std;
using namespace httplib;

const char *PAGINA = R"HTML(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Morfologia Matematica</title>
</head>
<body>
  <h2>Detecção de Fronteiras</h2>
  <input type="file" id="arquivo" accept="image/*">
  <button onclick="enviar()">Processar</button>
  <p id="status"></p>

  <div id="painel" hidden>
    <h3>Folhas individuais</h3>
    <button onclick="mudar(-1)">Anterior</button>
    <span id="info"></span>
    <button onclick="mudar(1)">Próxima</button><br><br>
    <img id="imgFolha" style="max-width:800px; border:1px solid #000">

    <h3>Todas as fronteiras</h3><span>Neste caso, são mostradas as manchas também</span>
    <img id="imgGeral" style="max-width:800px; border:1px solid #000">
  </div>
 
  <script>
    let pixels = []; // pixels de fronteira de cada folha
    let atual = 0; // folha mostrada
    let versao = 0;

    async function enviar() {
      const f = document.getElementById('arquivo').files[0];
      
      if (!f){
        return;
      }

      document.getElementById('painel').hidden = true;

      const resp = await fetch('/processar',
      { 
        method: 'POST',
        body: f 
      });

      const texto = (await resp.text()).trim();
      if (!resp.ok) {
        document.getElementById('status').textContent = texto;
        return;
      }

      // Responde com os pixels de cada folha separados por espaço (1234 987 1500)
      if(texto == ''){
        pixels = [];
      } else{
        pixels = texto.split(' ').map(Number);
      }
      
      atual = 0;
      versao++;

      document.getElementById('status').textContent = pixels.length + ' folha(s) encontrada(s)';
      if (pixels.length == 0){
        return;
      }

      document.getElementById('imgGeral').src = '/geral?v=' + versao;
      mostrarFolha();
      document.getElementById('painel').hidden = false;
    }

    function mostrarFolha() {
      document.getElementById('info').textContent =
        'Folha ' + (atual + 1) + ' de ' + pixels.length + ': ' +
        pixels[atual].toLocaleString('pt-BR') + ' pixels de fronteira';
      document.getElementById('imgFolha').src = '/folha?i=' + atual + '&v=' + versao;
    }

    function mudar(passo) {
      atual = (atual + passo + pixels.length) % pixels.length;
      mostrarFolha();
    }
  </script>
</body>
</html>
)HTML";

Mat bin(Mat img){
  Mat img_bin(img.rows, img.cols, CV_8UC1);
  for (int i = 0; i < img.rows; i++){
    for (int j = 0; j < img.cols; j++){
      Vec3b pixel = img.at<Vec3b>(i, j);
      int media = (pixel[0] + pixel[1] + pixel[2]) / 3;

      if (media < 128){
        img_bin.at<uchar>(i, j) = 0;
      } else {
        img_bin.at<uchar>(i, j) = 255;
      }
    }
  }
  return img_bin;
}

Mat pad(Mat img_bin){
  Mat img_pad(img_bin.rows+2, img_bin.cols+2, CV_8UC1, Scalar(255));
  for (int i = 0; i < img_bin.rows; i++){
    for (int j = 0; j < img_bin.cols; j++){
      img_pad.at<uchar>(i+1, j+1) = img_bin.at<uchar>(i, j);
    }
  }
  return img_pad;
}

Mat erosao(Mat img_pad){
  Mat img_erosao(img_pad.rows, img_pad.cols, CV_8UC1);
  for (int i = 0; i < img_pad.rows; i++){
    for (int j = 0; j < img_pad.cols; j++){
      img_erosao.at<uchar>(i, j) = img_pad.at<uchar>(i, j);
    }
  }

  // Erosão
  for (int i = 1; i < img_pad.rows - 1; i++){
    for (int j = 1; j < img_pad.cols - 1; j++){
      // verifica se o pixel central é preto, ou seja, objeto
      bool flag = false;
      if (img_pad.at<uchar>(i, j) == 0){
        for (int k = -1; k <= 1 && flag == false; k++){
          for (int l = -1; l <= 1 && flag == false; l++){
            if (img_pad.at<uchar>(i + k, j + l) == 255){
              // se algum pixel vizinho for branco, o pixel central é definido como branco
              img_erosao.at<uchar>(i, j) = 255;
              flag = true;
            }
          }
        }
      }
    }
  }
  return img_erosao;
}

Mat borda(Mat img_erosao, Mat img_pad){
  Mat img_resultado(img_erosao.rows-2, img_erosao.cols-2, CV_8UC3, Scalar(255, 255, 255));
  // Destaca a borda do objeto
  for (int i = 1; i < img_erosao.rows - 1; i++){
    for (int j = 1; j < img_erosao.cols - 1; j++){
      if (img_erosao.at<uchar>(i, j) == 255 && img_pad.at<uchar>(i, j) == 0){
        img_resultado.at<Vec3b>(i-1, j-1) = Vec3b(0, 0, 255); // Destaca a borda do objeto em vermelho
      }
    }
  }
  return img_resultado;
}

// Guarda 1 = fronteira e 0 = resto
Mat mascaraFronteira(Mat img_erosao, Mat img_pad){
  Mat img_fronteira(img_pad.rows, img_pad.cols, CV_8UC1, Scalar(0));
  for (int i = 1; i < img_pad.rows - 1; i++){
    for (int j = 1; j < img_pad.cols - 1; j++){
      if (img_erosao.at<uchar>(i, j) == 255 && img_pad.at<uchar>(i, j) == 0){
        img_fronteira.at<uchar>(i, j) = 1;
      }
    }
  }
  return img_fronteira;
}

// Os 8 vizinhos de um pixel em sentido horário, começando pelo oeste (W):
//   NW  N  NE        1 2 3
//   W   *  E    ->   0 * 4
//   SW  S  SE        7 6 5

const int VIZ_LINHA[8]  = { 0, -1, -1, -1, 0, 1, 1,  1};
const int VIZ_COLUNA[8] = {-1, -1,  0,  1, 1, 1, 0, -1};

// Passos 3 e 4 livro
bool proximoPonto(const Mat& img_fronteira, Point& b, Point& c){
  int d = 0;  // posição de c na lista de vizinhos de b
  while (b.y + VIZ_LINHA[d] != c.y || b.x + VIZ_COLUNA[d] != c.x) d++;
  for (int k = 1; k <= 8; k++){
    const int v = (d + k) % 8;
    const Point n(b.x + VIZ_COLUNA[v], b.y + VIZ_LINHA[v]);
    if (img_fronteira.at<uchar>(n) == 1){
      const int anterior = (d + k - 1) % 8;
      c = Point(b.x + VIZ_COLUNA[anterior], b.y + VIZ_LINHA[anterior]);
      b = n;
      return true;
    }
  }
  return false;
}

vector<Point> seguirFronteira(const Mat& img_fronteira, Point b0){
  vector<Point> pontos;
  pontos.push_back(b0);

  // Passos 1 e 2 livro
  Point b = b0;
  Point c(b0.x - 1, b0.y);
  proximoPonto(img_fronteira, b, c);
  const Point b1 = b;

  // Passos 3 a 5 livro
  while (true){
    pontos.push_back(b);
    const Point anterior = b;
    proximoPonto(img_fronteira, b, c);
    if (anterior == b0 && b == b1) break;
  }
  return pontos;
}

struct Folha{
  int area;
  vector<Point> pontos;
  int pixelsFronteira;
};

// Marca os pixels da folha em "visitado" e devolve quantos são
int marcarFolha(const Mat& img_pad, Mat& visitado, Point p){
  vector<Point> pilha;
  pilha.push_back(p);
  visitado.at<uchar>(p) = 1;
  int area = 0;
  while (!pilha.empty()){
    const Point q = pilha.back();
    pilha.pop_back();
    area++;
    for (int v = 0; v < 8; v++){
      const Point n(q.x + VIZ_COLUNA[v], q.y + VIZ_LINHA[v]);
      if (img_pad.at<uchar>(n) == 0 && visitado.at<uchar>(n) == 0){
        visitado.at<uchar>(n) = 1;
        pilha.push_back(n);
      }
    }
  }
  return area;
}

// Manchas menores que isso são tratadas como ruído e não como folha
const double FRACAO_MINIMA_FOLHA = 0.001;

// Encontra as folhas, segue a fronteira e conta os pixels dela
vector<Folha> analisarFolhas(const Mat& img_pad, const Mat& img_fronteira, int& ignorados){
  Mat visitado(img_pad.rows, img_pad.cols, CV_8UC1, Scalar(0));
  Mat contado(img_pad.rows, img_pad.cols, CV_8UC1, Scalar(0));
  const double areaMinima = FRACAO_MINIMA_FOLHA * (img_pad.rows - 2) * (img_pad.cols - 2);
  vector<Folha> folhas;
  ignorados = 0;

  for (int i = 1; i < img_pad.rows - 1; i++){
    for (int j = 1; j < img_pad.cols - 1; j++){
      if (img_pad.at<uchar>(i, j) != 0 || visitado.at<uchar>(i, j) == 1) continue;

      const Point b0(j, i);
      Folha f;
      f.area = marcarFolha(img_pad, visitado, b0);
      if (f.area < areaMinima){
        ignorados++;
        continue;
      }

      f.pontos = seguirFronteira(img_fronteira, b0);
      f.pixelsFronteira = 0;
      for (const Point& p : f.pontos){
        if (contado.at<uchar>(p) == 0){
          contado.at<uchar>(p) = 1;
          f.pixelsFronteira++;
        }
      }
      folhas.push_back(f);
    }
  }
  return folhas;
}

Mat imagemFolha(const Folha& f, int linhas, int colunas){
  Mat img(linhas, colunas, CV_8UC3, Scalar(255, 255, 255));
  for (const Point& p : f.pontos){
    img.at<Vec3b>(p.y - 1, p.x - 1) = Vec3b(0, 0, 255);
  }
  return img;
}

string codificarPNG(const Mat& img){
  vector<uchar> buf;
  imencode(".png", img, buf);
  return string(buf.begin(), buf.end());
}

vector<Folha> folhasAtuais;
int linhasAtuais = 0, colunasAtuais = 0;
string pngGeral;

int main()
{
  Server srv;

  srv.Get("/", [](const Request &, Response &res){
    res.set_content(PAGINA, "text/html");
  });

  srv.Post("/processar", [](const Request &req, Response &res){
    vector<uchar> dados(req.body.begin(), req.body.end());
    Mat img = imdecode(dados, IMREAD_COLOR);

    if (img.empty()){
      res.status = 400;
      res.set_content("Imagem invalida", "text/plain");
      return;
    }

    Mat img_bin = bin(img);
    Mat img_pad = pad(img_bin);
    Mat img_erosao = erosao(img_pad);
    Mat img_resultado = borda(img_erosao, img_pad);

    Mat img_fronteira = mascaraFronteira(img_erosao, img_pad);
    int ignorados = 0;
    vector<Folha> folhas = analisarFolhas(img_pad, img_fronteira, ignorados);
    cout << "Imagem " << img.cols << " x " << img.rows << ": " << folhas.size() << " folha(s)" << endl;
    for (size_t k = 0; k < folhas.size(); k++){
      cout << "  Folha " << k + 1 << ": " << folhas[k].pixelsFronteira << " pixels de fronteira" << endl;
    }
    if (ignorados > 0){
      cout << "  (" << ignorados << " mancha(s) ignorada(s) como ruido)" << endl;
    }

    folhasAtuais = folhas;
    linhasAtuais = img.rows;
    colunasAtuais = img.cols;
    vector<uchar> buf;
    imencode(".png", img_resultado, buf);
    pngGeral = string(buf.begin(), buf.end());

    string resposta;
    for (size_t k = 0; k < folhas.size(); k++){
      if (k > 0) resposta += " ";
      resposta += to_string(folhas[k].pixelsFronteira);
    }
    res.set_content(resposta, "text/plain");
  });

  srv.Get("/folha", [](const Request &req, Response &res){
    int i = atoi(req.get_param_value("i").c_str());
    
    Mat img = imagemFolha(folhasAtuais[i], linhasAtuais, colunasAtuais);
    res.set_content(codificarPNG(img), "image/png");
  });

  srv.Get("/geral", [](const Request &, Response &res){
    res.set_content(pngGeral, "image/png");
  });

  cout << "Servidor rodando em http://localhost:8080" << endl;
  srv.listen("0.0.0.0", 8080);

}
