from shop.refrigerator import Refrigerator
from shop.barista import Barista

if __name__ == '__main__':
    refrigerator = Refrigerator()
    refrigerator.add_beans("하와이 코다", 5)
    refrigerator.add_beans("케냐AA", 2)
    refrigerator.print_inventory()

    barista = Barista(name="카누", available_menus=["에스프레소", "아메리카노"])
    coffee = barista.receive_order(refrigerator)

    if coffee is None:
        print("커피를 받지 못했습니다.")
    else:
        coffee.print_info()
