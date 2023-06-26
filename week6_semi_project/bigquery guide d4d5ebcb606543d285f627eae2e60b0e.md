# bigquery guide

### 빅쿼리

google cloud platform에서 제공하는 data warehouse 솔루션으로 대량의 데이터를 SQL을 이용하여 분석할 수 있는 솔루션. 실제로 많은 스타트업들이 사용함.

### 사용법

다양한 사용법이 있지만, 이 가이드에서는 CSV 파일을 google cloud storage에 업로드하고, 이를 bigquery에서 import 한 뒤, SQL을 돌려보는 것을 목적으로 함.

### 사용할 데이터 셋

데이터 베이스 수업 시간에 다뤘던 olist 데이터 셋의 orders.csv 파일을 가지고 가이드를 진행해보겠습니다.

### google cloud platform에 로그인

- 구글 아이디가 있으면 바로 로그인 가능
- 결제 정보를 등록해야 솔루션들을 사용 가능
- 결제 정보 등록하면 300달러 크레딧을 주는데, 이 크레딧을 다 사용하기 전까지는 결제 일어나지 않음.

### google cloud storage에 업로드

- google cloud storage는 일종의 구글 드라이브라고 생각하면 됨.
- 빅쿼리 테이블에 데이터를 올려놓기 전에 파일 형태의 데이터는 먼저 google cloud storage에 올려놓으면 처리하기가 편함.

![Untitled](bigquery%20guide%20d4d5ebcb606543d285f627eae2e60b0e/Untitled.png)

![Untitled](bigquery%20guide%20d4d5ebcb606543d285f627eae2e60b0e/Untitled%201.png)

- Region - 서울로 지정하고 나머지는 기본 값으로 놓고 버킷 생성. 버킷은 구글 드라이브 안에 일종의 폴더라고 생각하면 됨.

![Untitled](bigquery%20guide%20d4d5ebcb606543d285f627eae2e60b0e/Untitled%202.png)

- 파일 업로드 버튼 누른 뒤, 데이터 파일 선택해서 업로드 진행

![Untitled](bigquery%20guide%20d4d5ebcb606543d285f627eae2e60b0e/Untitled%203.png)

### Bigquery에서 데이터 파일 import

- 데이터 추가 버튼을 누른 뒤, Google Cloud Storage 선택

![Untitled](bigquery%20guide%20d4d5ebcb606543d285f627eae2e60b0e/Untitled%204.png)

- 소스는 Google Cloud Storage를 선택하고 찾아보기 버튼을 눌러서 앞서 올려놓은 데이터 선택

![Untitled](bigquery%20guide%20d4d5ebcb606543d285f627eae2e60b0e/Untitled%205.png)

- 데이터 셋 ID 지정하고, 멀티 리전이 아닌 리전 선택 후 서울 지정한 다음 데이터 셋 만들어 줌

![Untitled](bigquery%20guide%20d4d5ebcb606543d285f627eae2e60b0e/Untitled%206.png)

- 스키마 자동 감지 켜준 뒤 테이블 만들기 시작

![Untitled](bigquery%20guide%20d4d5ebcb606543d285f627eae2e60b0e/Untitled%207.png)

- import 된 테이블 확인

![Untitled](bigquery%20guide%20d4d5ebcb606543d285f627eae2e60b0e/Untitled%208.png)

### Bigquery Console에서 Query 날리기

- 일자별 주문 건수 집계

![Untitled](bigquery%20guide%20d4d5ebcb606543d285f627eae2e60b0e/Untitled%209.png)