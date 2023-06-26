class Espresso:
    def __init__(self, bean, shots):
        print("espresso 객체를 생성합니다.")
        self.bean = bean
        self.shots = shots

    def print_espresso(self):
        print("에스프레소 정보")
        print("bean", self.bean if self.bean else "지정 안함")
        print("shot", self.shots if self.shots else "지정 안함")
        print("-----------------------")

    def print_info(self):
        self.print_espresso()


class Americano(Espresso):
    def __init__(self, bean, shots, is_hot, size):
        print("americano 객체를 생성합니다.")
        self.bean = bean
        self.shots = shots
        self.is_hot = is_hot
        self.size = size

    def print_americano(self):
        print("아메리카노 정보")
        print("bean", self.bean)
        print("shots", self.shots)
        print("is_hot", self.is_hot)
        print("size", self.size)
        print("-----------------------")

    def print_info(self):
        self.print_americano()
