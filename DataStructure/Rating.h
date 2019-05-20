#ifndef RATING_H
#define RATING_H

#include <stdio.h>
#include <iostream>
#include <string>

struct Rating {
	int userId;
	int itemId; 
	float score;
	long timestamp;

	Rating() = default;
	
	Rating(int _userId,
		int _itemId,
		float _score,
		long _timestamp) :
		userId(_userId),
		itemId(_itemId),
		score(_score),
		timestamp(_timestamp) { }
};
//bool LessSort(Rating a, Rating b) { return(a.timestamp < b.timestamp); }


#endif // !RATING_H

