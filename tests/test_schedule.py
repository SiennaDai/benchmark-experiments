import pytest
from train_platform import learning_rate

def test_schedule_boundaries():
    assert learning_rate(1,10,1,2,.1,"cosine")==.5 and learning_rate(2,10,1,2,.1,"cosine")==1
    assert learning_rate(10,10,1,2,.1,"cosine")==pytest.approx(.1) and learning_rate(1,1,2,0,.1,"cosine")==2
    assert learning_rate(1,4,1,0,.2,"cosine")==1 and learning_rate(4,4,1,0,.2,"cosine")==pytest.approx(.2)
