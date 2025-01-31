
cards = dict()
cards['Sports'] = ['Football', 'Soccer', 'Golf', 'Baseball', 'Basketball', 'Ice Hockey', 'Sailing', 'Squash', 'Badminton', 'Tennis', 'Wrestling', 'Rock Climbing', 'Motor Racing', 'Triathlon', 'Volleyball', 'Cycling']

for category in cards.keys():
    for i in range(len(cards[category])):
        cards[category][i] = cards[category][i].lower()

#Save the cards in a pickle file
import pickle
with open('ChameleonCards.pkl', 'wb') as f:
    pickle.dump(cards, f)

#Load the cards from the pickle file
import pickle
with open('ChameleonCards.pkl', 'rb') as f:
    cards = pickle.load(f)



