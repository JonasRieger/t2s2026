# Following Socio-Environmental Conflict Narratives About Energy Transition in Chile
### A Spatio-Temporal Analysis Using Dynamic Topic Modeling

[Dashboard](https://rawcdn.githack.com/JonasRieger/t2s2026/9030bbf0f823a9781c42135e0f2deeeeaa8d86f9/board.html).

This repository provides some data and scripts related to the paper:

* Rieger, J., Muñoz, F., Grönberg, L., Lange, K.-R., Ojeda-Pereira, I., Briceño, D., Nass, C., Stahl, C., Cassola, J., Rojas-Córdova, C., Keith-Norambuena, B., Lufin, M., Campos-Medina, F., Herrera-León, S. (2026). Following socio-environmental conflict narratives about energy transition in Chile: A spatio-temporal analysis using dynamic topic modeling. Accepted for [Text2Story’26 Workshop](https://text2story26.inesctec.pt/).

For bug reports, comments and questions please use the [issue tracker](https://github.com/JonasRieger/t2s2026/issues).

Please note: For legal reasons the repository cannot provide all data, e.g., the scraped data are deleted. Please let us know if you feel that there is anything missing that we could add.


## Development Setup

To start the development setup, we suggest to install the small package as an editable package. This allows for easier imports for the constructed functions. With the installation of an editable package, the package contents can also be changed without reinstallation during development.

### Package Installation
To install the package do the following:

1. Go to the root of the project
2. Execute `pip install --editable .` in the terminal, which installs the package to get an editable installation
3. Use the package via `import cet`, where cet stands for chilean energy transition.
4. Download the spacy models via:
   1. python -m spacy download es_core_news_sm
   2. python -m spacy download es_core_news_lg
   3. python -m spacy download es_core_news_md
      For the small, large and medium spacy preprocessing models.
5. Create a secrets.json file in the root of the project with the content of the secrets_example.json file.

### Configuration File

The follwoing defines the current **config.json** settings. Please see the _config.json_ file for the current definition.

```json
{
  "model_save_path": "data/models", // basepath where model results should be stored.
  "start_date_data": "2011-01-01", // start date for the analysis and preprocessing
  "num_workers": 6, // number of processes to spawn
  "spacy_pipeline": "es_core_news_lg", // spacy pipeline to use
  "words_to_exclude": [] // list contains words to exclude
}
```
