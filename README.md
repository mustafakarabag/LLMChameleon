# LLMChameleon
Repository for the paper "Do LLMs Strategically Reveal, Conceal, and Infer Information?  A Theoretical and Empirical Analysis in The Chameleon Game" [Paper link]


The Chameleon is a commercial board game. Please buy the game at bigpotato.com/products/the-chameleon . 
We provide a single game card to demonstrate the gameplay with LLMs. You can populate ChameleonCards.py with additional cards.

To make different LLMs play The Chameleon, run TestLlms.py . Note that you need API keys for different LLMs. Fill the api_keys.txt with the respective API keys of LLMs. PlotWinRatios.py plots different statistics of the games.

PlotInfoGains.py generates a figure like Figure 7 of the paper. You first need to compute the posterior probabilities by running the ComputeInfoGains.py. Note that ComputeInfoGains.py requires a Google Custom Search Engine ID and a Google Search API key.

