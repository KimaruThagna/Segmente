'''
    Treatment Responders(TR): Customers that will purchase only if they receive an offer
    Treatment Non-Responders(TN): Customer that won’t purchase in any case
    Control Responders(CR): Customers that will purchase without an offer(IF YOU OFFER, YOU WILL BE CANNIBALIZING)
    Control Non-Responders(CN): Customers that will not purchase if they don’t receive an offer

'''
## Uplift formula: Uplift score= P(TR)+P(CN)-P(TN)-P(CR)