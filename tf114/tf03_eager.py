import tensorflow as tf
print(tf.__version__)   # 1.14.0
print(tf.executing_eagerly())   # False

# 즉시실행모드 => 텐서플로우1의 그래프 형태의 구성없이 자연스러운 파이썬 문법으로 실행시킨다.
# tf.compat.v1.enable_eager_execution()   # 즉시실행모드 킨다. // 텐서플로우 2.0 사용가능
tf.compat.v1.disable_eager_execution()  # 즉시실행모드 끈다. // 텐서플로우 1.0 문법 // 디폴트
print(tf.executing_eagerly())   # True

hello = tf.constant('hello world')

sess = tf.compat.v1.Session()

print(sess.run(hello))
'''
Tensor1 은 '그래프연산' 모드
Tensor2 는 '즉시실행' 모드

tf.compat.v1.enable_eager_execution() : 즉시실행 모드 켜(tensor2 디폴트)
tf.compat.v1.disable_eager_execution() : 즉시실행 모드 꺼(tensor1 디폴트) -> 그래프 연산모드로 돌아간다 -> Tensor1 코드를 쓸 수 있다.

tf.executing_eagerly() :    True면 즉시 실행모드 -> tensor2 코드만 써야한다.
                            False면 그래프 연산모드 ->tensor1 코드를 쓸 수 있다.
'''