# Simple-RAG

OllamaでLLMを、ChainlitでUIを、LangchainでRAGを構築したやつ。

## サンプル

- NTTと野村総研、三菱電機のWikipedia情報をMarkdown/PDF/HTML形式で入力している

![ntt](./images/ntt.png)
![nri](./images/nri.png)
![melco](./images/melco.png)

## 手順

- VSCodeで本リポジトリをdevcontainerで起動する
- ターミナルで、`chainlit run app.py`を実行
  - ドキュメントをベクトル化するので、だいぶ遅い
